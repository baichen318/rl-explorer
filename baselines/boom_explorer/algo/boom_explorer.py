# Author: baichen318@gmail.com


import os
import tqdm
import torch
import random
import gpytorch
import functools
import numpy as np
from time import time
from datetime import datetime
try:
    from sklearn.externals import joblib
except ImportError:
    import joblib
from utils.utils import info, mkdir
from baselines.boom_explorer.algo.dkl_gp import DKLGP
from botorch.utils.multi_objective.hypervolume import Hypervolume
from botorch.utils.multi_objective.pareto import is_non_dominated
from simulation.boom.simulation import Gem5Wrapper as BOOMGem5Wrapper
from simulation.rocket.simulation import Gem5Wrapper as RocketGem5Wrapper
from botorch.acquisition.multi_objective.analytic import ExpectedHypervolumeImprovement
from baselines.boom_explorer.util.sample import micro_al, random_sample, initial_random_sample
from botorch.utils.multi_objective.box_decompositions.non_dominated import NondominatedPartitioning
from baselines.boom_explorer.util.util import calc_adrs, array_to_tensor, tensor_to_array, scale_dataset, \
    rescale_dataset


def get_pareto_set(y: torch.Tensor):
    """
        NOTICE: `is_non_dominated` assumes maximization
    """
    return y[is_non_dominated(y)]


def get_pareto_optimal_solutions(x: torch.Tensor, y: torch.Tensor):
    return x[is_non_dominated(y)].long()


def initialize_dkl_gp(configs, x, y):
    return DKLGP(configs, x, y, mlp_output_dim=configs["mlp-output-dim"])


def fit_dkl_gp(configs, x, y):
    # NOTICE: Because the dataset `x` is enlarged in every
    # iteration, according to GP, new GP are built w.r.t `x`.
    # Therefore, new GP is trained in every iteration.
    model = initialize_dkl_gp(configs, x, y)
    model.set_train()

    parameters = [
        {"params": model.mlp.parameters()},
        {"params": model.gp.covar_module.parameters()},
        {"params": model.gp.mean_module.parameters()},
        {"params": model.gp.likelihood.parameters()}
    ]
    optimizer = torch.optim.Adam(parameters, lr=configs["learning-rate"])

    iterator = tqdm.trange(configs["max-optimize-epoch"], desc="Training DKL-GP")
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


def ehvi_suggest(model, problem, sampled_x, sampled_y, batch=1):
    """
        model: <DKLGP>
        problem: <MultiObjectiveTestProblem>
        sampled_x: <torch.Tensor>
        sampled_y: <torch.Tensor>
    """
    partitioning = NondominatedPartitioning(
        ref_point=problem._ref_point.to(model.device),
        Y=sampled_y.to(model.device)
    )

    acq_func = ExpectedHypervolumeImprovement(
        model=model.gp,
        ref_point=problem._ref_point.tolist(),
        partitioning=partitioning
    ).to(model.device)

    acq_val = acq_func(
        model.forward_mlp(
            sampled_x.to(torch.float).to(model.device)
        ).unsqueeze(1).to(model.device)
    ).to(model.device)
    top_acq_val, indices = torch.topk(acq_val, k=batch)
    new_x = sampled_x[indices].to(torch.float32)
    del acq_val
    return new_x.reshape(-1, problem.n_dim), torch.mean(top_acq_val)


def report(
    configs,
    search_x,
    pareto_frontier,
    pareto_optimal_solutions,
    ort
):
    # write results
    # create the result directory
    result_root = os.path.join(os.path.dirname(
        os.path.abspath(__file__)), "result"
    )
    mkdir(result_root)

    with open(os.path.join(
            result_root, "{}.txt".format(datetime.now()).replace(' ', '-')
        ), 'w'
    ) as f:
        n_samples = pareto_optimal_solutions.shape[0]
        f.write(
            "obtained solution: {}:\n".format(
                n_samples
            )
        )
        for i in range(n_samples):
            f.write(
                "{}\t{}\n".format(
                    pareto_optimal_solutions[i].int().tolist(),
                    pareto_frontier[i].float().tolist()
                )
            )
        f.write("microarchitecture (follow the order):\n")
        for x in search_x:
            f.write("{}\n".format(x[0].int().tolist()))
        f.write('\n')
        f.write("cost time: {}s.".format(ort))


def evaluate_microarchitecture(configs, design_space, embedding, boom):
    def load_ppa_model():
        ppa_model_root = os.path.join(
            configs["rl-explorer-root"],
            configs["env"]["calib"]["ppa-model"]
        )
        perf_root = os.path.join(
            ppa_model_root,
            "boom-perf.pt"
        )
        power_root = os.path.join(
            ppa_model_root,
            "boom-power.pt"
        )
        area_root = os.path.join(
            ppa_model_root,
            "boom-area.pt"
        )
        perf_model = joblib.load(perf_root)
        power_model = joblib.load(power_root)
        area_model = joblib.load(area_root)
        return perf_model, power_model, area_model

    perf_model, power_model, area_model = load_ppa_model()
    if boom:
        Simulator = BOOMGem5Wrapper
    else:
        Simulator = RocketGem5Wrapper
    manager = Simulator(
        configs,
        design_space,
        embedding,
        configs["env"]["sim"]["idx"]
    )
    perf, stats = manager.evaluate_perf()
    power, area = manager.evaluate_power_and_area()
    area *= 1e6
    stats_feature = []
    for k, v in stats.items():
        stats_feature.append(v)
    stats_feature = np.array(stats_feature)
    perf = perf_model.predict(np.expand_dims(
            np.concatenate((
                    np.array(embedding),
                    stats_feature,
                    [perf]
                )
            ),
            axis=0
            )
        )[0]
    power = power_model.predict(np.expand_dims(
            np.concatenate((
                    np.array(embedding),
                    stats_feature,
                    [power]
                )
            ),
            axis=0
            )
        )[0]
    area = area_model.predict(np.expand_dims(
            np.concatenate((
                    np.array(embedding),
                    stats_feature,
                    [area]
                )
            ),
            axis=0
            )
        )[0]
    area *= 1e-6
    return scale_dataset(
        array_to_tensor(
            np.array([perf, power, area])
        ).unsqueeze(0),
        boom
    )


def boom_explorer(configs, settings, problem):

    # set random seed
    random.seed(settings["seed"])
    np.random.seed(settings["seed"])
    torch.manual_seed(settings["seed"])

    hv = Hypervolume(ref_point=problem._ref_point)
    # adrs saves ADRS before and after running of BOOM-Explorer
    adrs = []

    start = time()
    # generate initial data
    info("sampling...")
    if problem.boom:
        x, y = micro_al(settings, problem)
    else:
        x, y = initial_random_sample(settings, problem, settings["batch"])
    adrs.append(
        calc_adrs(
            get_pareto_set(problem.total_y),
            get_pareto_set(y)
        )
    )

    info("initialize DKL-GP...")
    model = initialize_dkl_gp(settings, x, y)

    # Bayesian optimization
    search_x, search_acq_val = [], []
    iterator = tqdm.tqdm(range(settings["max-bo-steps"]))
    for step in iterator:
        iterator.set_description("Iter %d" % (step + 1))
        # train
        model = fit_dkl_gp(settings, x, y)
        # sample by acquisition function
        _x = array_to_tensor(random_sample(
                configs,
                problem,
                batch=settings["sample-design-space-size"]
            )
        )
        new_x, acq_val = ehvi_suggest(model, problem, _x, y)
        search_acq_val.append(acq_val)
        # add in to `_x` and `_y`
        search_x.append(new_x)
        x = torch.cat((x, new_x), 0)
        y = torch.cat(
            (
                y,
                evaluate_microarchitecture(
                    configs,
                    problem.design_space,
                    tensor_to_array(new_x).astype("int").ravel(),
                    problem.boom
                )
            ),
            0
        )

    pareto_frontier = get_pareto_set(y)
    adrs.append(
        calc_adrs(
            get_pareto_set(problem.total_y),
            pareto_frontier
        )
    )
    end = time()
    ort = end - start
    info(
        "reference pareto frontier (no golden pareto frontier): \n{},\n" \
        "size: {}, ADRS: {}, ORT: {} s.".format(
            rescale_dataset(pareto_frontier, problem.boom),
            len(pareto_frontier),
            adrs,
            ort
        )
    )

    # report
    report(
        configs,
        search_x,
        pareto_frontier,
        get_pareto_optimal_solutions(x, y),
        ort
    )
    info("BOOM-Explorer done.")
