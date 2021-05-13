# Author: baichen318@gmail.com

import os
import torch
import gpytorch
import numpy as np
from sample import ClusteringRandomizedTED
from dnn_gp import DNNGP
from vis import validate
from util import parse_args, get_configs, execute, read_csv_v2, mse, r2, mape
from exception import UnDefinedException

def get_data():
    dataset, title = read_csv_v2(configs["dataset-output-path"])
    dataset = validate(dataset)
    return dataset

def split_data(data, x, y):
    """
        data: <numpy.ndarray>
        x: <list>
        y: <list>
    """
    def _split(data):
        for d in data:
            # In order to balance the c.c. and power, we scale them with suitable factors
            yield d[:-2], np.array([d[-2] / 10000, d[-1] * 100])
    for _x, _y in _split(data):
        x.append(_x)
        y.append(_y)
    return torch.tensor(x).float(), torch.tensor(y).float()

def construct_test_dataset(dataset):
    idx = [851, 328, 252, 612, 594, 379, 393,  53, 255, 872, 174,  12,  88,
       818, 771, 633, 505, 168,  96, 301, 363, 101, 664, 809, 203, 550,
       187, 399, 670, 353, 510, 722, 455, 577, 233, 496, 372, 733, 357,
       577, 814, 533, 753, 627, 307,  48, 639, 225, 841, 343, 271, 432,
       452, 525,  92,  22, 210, 777, 562,  82, 162, 723, 427, 795, 135,
       373, 600, 651, 397, 188,  34, 128, 564, 337, 582, 408, 327, 730,
        53, 116, 297, 498, 101, 211,  18, 258, 767, 180, 476, 161,  73,
       160, 472, 679, 886, 311, 393, 814, 637, 111]

    test_dataset = []
    for i in idx:
        test_dataset.append(dataset[i])

    x, y = [], []
    x, y = split_data(np.array(test_dataset), x, y)
    # remove `test_dataset` from `dataset`
    dataset = np.delete(dataset, idx, axis=0)

    return [x, y], dataset

def initialize_dnn_gp(x, y):
    """
        x: <torch.Tensor>
        y: <torch.Tensor>
    """
    model = DNNGP(x, y, configs)
    return model

def fit_dnn_gp(x, y):
    model = DNNGP(x, y, configs)
    model.set_train()

    params = [
        {"params": model.mlp.parameters()},
        {"params": model.gp.covar_module.parameters()},
        {"params": model.gp.mean_module.parameters()},
        {"params": model.gp.likelihood.parameters()}
    ]
    if configs["optimizer"] == "Adam":
        optimizer = torch.optim.Adam(params, lr=configs["learning-rate"])
    elif configs["optimizer"] == "SGD":
        optimizer = torch.optim.SGD(params, lr=configs["learning-rate"])
    else:
        raise UnDefinedException(configs["optimizer"])

    mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.gp.likelihood, model.gp)
    y = model.transform_ylayout(y).squeeze(1).to(model.device)

    for i in range(configs["max-epoch"]):
        optimizer.zero_grad()
        _y = model.forward(x)
        loss = -mll(_y, y)
        loss.backward()
        optimizer.step()
        if (i + 1) % 20 == 0:
            print("[INFO]: iter: %d\tloss: %.8f" % ((i + 1), loss))
    model.set_eval()

    return model

def sample_by_acquisition(model):
    pass

def predict_by_dnn_gp(model, x, y):
    """
        Integrate metrics analysis
    """
    def analysis(y1, y2):
        mse_l = mse(y1[:, 0], y2[:, 0])
        mse_p = mse(y1[:, 1], y2[:, 1])
        r2_l = r2(y1[:, 0], y2[:, 0])
        r2_p = r2(y1[:, 1], y2[:, 1])
        mape_l = mape(y1[:, 0], y2[:, 0])
        mape_p = mape(y1[:, 1], y2[:, 1])

        return mse_l, mse_p, r2_l, r2_p, mape_l, mape_p

    _y = model.predict(x)
    y = y.to(model.device)
    return analysis(_y.numpy(), y.numpy())

def sample(sampler, unsampled_dataset, sampled_dataset):
    data = sampler.crted(unsampled_dataset)
    # move sampled data from `unsampled_dataset` to `sampled_dataset`
    temp = []
    for d in data:
        idx = 0
        for _d in unsampled_dataset:
            if ((_d - d < 1e-5)).all():
                temp.append(idx)
                break
            idx += 1
        sampled_dataset.append(d)
    unsampled_dataset = np.delete(unsampled_dataset, temp, axis=0)
    return unsampled_dataset

def design_explorer():

    dataset = get_data()
    test_dataset, dataset = construct_test_dataset(dataset)
    unsampled_dataset = dataset.copy()
    sampled_dataset = []
    sampler = ClusteringRandomizedTED(configs)

    # initialize
    x, y = [], []
    unsampled_dataset = sample(sampler, unsampled_dataset, sampled_dataset)
    x, y = split_data(sampled_dataset, x, y)
    for i in range(configs["max-bo-steps"]):
        model = fit_dnn_gp(x, y)
        metrics = predict_by_dnn_gp(model, x, y)
        msg = "[TRAIN] MSE (latency): %.8f, MSE (power): %.8f, " % (metrics[0], metrics[1]) + \
            "MAPE (latency): %.8f, MAPE (power): %.8f, " % (metrics[4], metrics[5]) + \
            "R2 (latency): %.8f, R2 (power): %.8f, " % (metrics[2], metrics[3]) + \
            "training data size: %d" % len(sampled_dataset)
        print(msg)
        unsampled_dataset = sample(sampler, unsampled_dataset, sampled_dataset)
        x, y = split_data(sampled_dataset, x.tolist(), y.tolist())

        # evaluate
        metrics = predict_by_dnn_gp(model, test_dataset[0], test_dataset[1])
        msg = "[TEST] MSE (latency): %.8f, MSE (power): %.8f, " % (metrics[0], metrics[1]) + \
                "MAPE (latency): %.8f, MAPE (power): %.8f, " % (metrics[4], metrics[5]) + \
                "R2 (latency): %.8f, R2 (power): %.8f, " % (metrics[2], metrics[3]) + \
                "on test data"
        print(msg)

    # validate on `unsampled_dataset`
    x, y = [], []
    x, y = split_data(unsampled_dataset, x, y)
    metrics = predict_by_dnn_gp(model, x, y)
    msg = "[FINAL-TEST] MSE (latency): %.8f, MSE (power): %.8f, " % (metrics[0], metrics[1]) + \
            "MAPE (latency): %.8f, MAPE (power): %.8f, " % (metrics[4], metrics[5]) + \
            "R2 (latency): %.8f, R2 (power): %.8f, " % (metrics[2], metrics[3]) + \
            "unsampled_dataset data size: %d" % len(unsampled_dataset)
    print(msg)
    # validate on `test_dataset`
    metrics = predict_by_dnn_gp(model, test_dataset[0], test_dataset[1])
    msg = "[FINAL-TEST] MSE (latency): %.8f, MSE (power): %.8f, " % (metrics[0], metrics[1]) + \
            "MAPE (latency): %.8f, MAPE (power): %.8f, " % (metrics[4], metrics[5]) + \
            "R2 (latency): %.8f, R2 (power): %.8f, " % (metrics[2], metrics[3]) + \
            "on test data"
    print(msg)
    model.save(
        os.path.join(
            configs["model-output-path"],
            "dnn-gp.mdl"
        )
    )

if __name__ == "__main__":
    argv = parse_args()
    configs = get_configs(argv.configs)
    design_explorer()
