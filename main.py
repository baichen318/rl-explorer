# Author: baichen318@gmail.com

import os
import torch
import gpytorch
import numpy as np
from sample import ClusteringRandomizedTED
from dnn_gp import DNNGP, DNNGPV2
from vis import validate
import joblib
from util import parse_args, get_configs, execute, read_csv_v2, mse, r2, mape
from exception import UnDefinedException

def get_data():
    dataset, title = read_csv_v2(configs["dataset-output-path"])
    dataset = validate(dataset)

    # scale dataset to balance c.c. and power dissipation
    dataset[:, -2] = dataset[:, -2] / 10000
    dataset[:, -1] = dataset[:, -1] * 100

    return dataset

def split_data(data):
    """
        data: <numpy.ndarray>
        x: <list>
        y: <list>
    """
    def _split(data):
        for d in data:
            yield d[:-2], np.array([d[-2], d[-1]])
    x, y = [], []
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

    x, y = split_data(np.array(test_dataset))
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
    return analysis(y.numpy(), _y.numpy())

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

def design_explorer_v1():
    dataset = get_data()
    test_dataset, dataset = construct_test_dataset(dataset)
    unsampled_dataset = dataset.copy()
    sampled_dataset = []
    sampler = ClusteringRandomizedTED(configs)

    # initialize
    unsampled_dataset = sample(sampler, unsampled_dataset, sampled_dataset)
    x, y = split_data(sampled_dataset)
    # record R2
    for i in range(configs["max-bo-steps"]):
        model = fit_dnn_gp(x, y)
        metrics = predict_by_dnn_gp(model, x, y)
        msg = "[TRAIN] MSE (latency): %.8f, MSE (power): %.8f, " % (metrics[0], metrics[1]) + \
            "MAPE (latency): %.8f, MAPE (power): %.8f, " % (metrics[4], metrics[5]) + \
            "R2 (latency): %.8f, R2 (power): %.8f, " % (metrics[2], metrics[3]) + \
            "training data size: %d" % len(sampled_dataset)
        print(msg)
        unsampled_dataset = sample(sampler, unsampled_dataset, sampled_dataset)
        x, y = split_data(sampled_dataset)

        # evaluate
        metrics = predict_by_dnn_gp(model, test_dataset[0], test_dataset[1])
        msg = "[TEST] MSE (latency): %.8f, MSE (power): %.8f, " % (metrics[0], metrics[1]) + \
                "MAPE (latency): %.8f, MAPE (power): %.8f, " % (metrics[4], metrics[5]) + \
                "R2 (latency): %.8f, R2 (power): %.8f, " % (metrics[2], metrics[3]) + \
                "on test data"
        print(msg)

    # validate on `unsampled_dataset`
    x, y = split_data(unsampled_dataset)
    metrics = predict_by_dnn_gp(model, x, y)
    msg = "[ONE-MORE-TEST] MSE (latency): %.8f, MSE (power): %.8f, " % (metrics[0], metrics[1]) + \
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

def design_explorer_v2():
    dataset = get_data()
    test_dataset, dataset = construct_test_dataset(dataset)
    unsampled_dataset = dataset.copy()
    sampled_dataset = []
    sampler = ClusteringRandomizedTED(configs)

    def analysis(gt, predict):
        _mse = mse(gt, predict)
        _r2 = r2(gt, predict)
        _mape = mape(gt, predict)

        return _mse, _r2, _mape

    # initialize
    unsampled_dataset = sample(sampler, unsampled_dataset, sampled_dataset)
    x, y = split_data(sampled_dataset)
    for i in range(configs["max-bo-steps"]):
        model_l = DNNGPV2(configs, x, y[:, 0])
        model_p = DNNGPV2(configs, x, y[:, 1])
        model_l.fit(x, y[:, 0])
        model_p.fit(x, y[:, 1])

        _y_l = model_l.predict(x)
        _y_p = model_p.predict(x)

        metrics_l = analysis(y[:, 0], _y_l.cpu().detach().numpy())
        metrics_p = analysis(y[:, 1], _y_p.cpu().detach().numpy())

        msg = "[TRAIN] MSE (latency): %.8f, MSE (power): %.8f, " % (metrics_l[0], metrics_p[0]) + \
            "MAPE (latency): %.8f, MAPE (power): %.8f, " % (metrics_l[2], metrics_p[2]) + \
            "R2 (latency): %.8f, R2 (power): %.8f, " % (metrics_l[1], metrics_p[1]) + \
            "training data size: %d" % len(sampled_dataset)
        print(msg)
        unsampled_dataset = sample(sampler, unsampled_dataset, sampled_dataset)
        x, y = split_data(sampled_dataset)

        # evaluate
        _y_l = model_l.predict(test_dataset[0])
        _y_p = model_p.predict(test_dataset[0])

        metrics_l = analysis(test_dataset[0][:, 0], _y_l.cpu().detach().numpy())
        metrics_p = analysis(test_dataset[1][:, 1], _y_p.cpu().detach().numpy())

        msg = "[TEST] MSE (latency): %.8f, MSE (power): %.8f, " % (metrics_l[0], metrics_p[0]) + \
            "MAPE (latency): %.8f, MAPE (power): %.8f, " % (metrics_l[2], metrics_p[2]) + \
            "R2 (latency): %.8f, R2 (power): %.8f, " % (metrics_l[1], metrics_p[1]) + \
            "on test data set"
        print(msg)

    # validate on `unsampled_dataset`
    x, y = split_data(unsampled_dataset)
    _y_l = model_l.predict(x)
    _y_p = model_p.predict(x)

    metrics_l = analysis(y[:, 0], _y_l.cpu().detach().numpy())
    metrics_p = analysis(y[:, 1], _y_p.cpu().detach().numpy())
    msg = "[FINAL-TEST] MSE (latency): %.8f, MSE (power): %.8f, " % (metrics_l[0], metrics_p[0]) + \
        "MAPE (latency): %.8f, MAPE (power): %.8f, " % (metrics_l[2], metrics_p[2]) + \
        "R2 (latency): %.8f, R2 (power): %.8f, " % (metrics_l[1], metrics_p[1]) + \
        "unsampled_dataset data size: %d" % len(unsampled_dataset)
    print(msg)
    # model.save(
    #     os.path.join(
    #         configs["model-output-path"],
    #         "dnn-gp.mdl"
    #     )
    # )

def design_explorer_v3():
    dataset = get_data()
    test_dataset, dataset = construct_test_dataset(dataset)

    def analysis(gt, predict):
        _mse = mse(gt, predict)
        _r2 = r2(gt, predict)
        _mape = mape(gt, predict)

        return _mse, _r2, _mape

    # initialize
    x, y = split_data(dataset)
    model_l = DNNGPV2(configs, x, y[:, 0])
    model_p = DNNGPV2(configs, x, y[:, 1])
    model_l.fit(x, y[:, 0])
    model_p.fit(x, y[:, 1])

    _y_l = model_l.predict(x)
    _y_p = model_p.predict(x)

    metrics_l = analysis(y[:, 0], _y_l.cpu().detach().numpy())
    metrics_p = analysis(y[:, 1], _y_p.cpu().detach().numpy())

    msg = "[TRAIN] MSE (latency): %.8f, MSE (power): %.8f, " % (metrics_l[0], metrics_p[0]) + \
        "MAPE (latency): %.8f, MAPE (power): %.8f, " % (metrics_l[2], metrics_p[2]) + \
        "R2 (latency): %.8f, R2 (power): %.8f, " % (metrics_l[1], metrics_p[1]) + \
        "training data size: %d" % len(sampled_dataset)
    print(msg)
    unsampled_dataset = sample(sampler, unsampled_dataset, sampled_dataset)
    x, y = split_data(sampled_dataset)

    # evaluate
    _y_l = model_l.predict(test_dataset[0])
    _y_p = model_p.predict(test_dataset[0])

    metrics_l = analysis(test_dataset[0][:, 0], _y_l.cpu().detach().numpy())
    metrics_p = analysis(test_dataset[1][:, 1], _y_p.cpu().detach().numpy())

    msg = "[TEST] MSE (latency): %.8f, MSE (power): %.8f, " % (metrics_l[0], metrics_p[0]) + \
        "MAPE (latency): %.8f, MAPE (power): %.8f, " % (metrics_l[2], metrics_p[2]) + \
        "R2 (latency): %.8f, R2 (power): %.8f, " % (metrics_l[1], metrics_p[1]) + \
        "on test data set"
    print(msg)

    # model.save(
    #     os.path.join(
    #         configs["model-output-path"],
    #         "dnn-gp.mdl"
    #     )
    # )

def analysis():
    import matplotlib.pyplot as plt
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import PolynomialFeatures

    markers = [
        '.', ',', 'o', 'v', '^', '<', '>', '1', '2', '3',
        '4', '8', 's', 'p', 'P', '*', 'h', 'H', '+', 'x',
        'X', 'D', 'd', '|', '_'
    ]
    colors = [
        'c', 'b', 'g', 'r', 'm', 'y', 'k', 'w'
    ]

    dataset = get_data()
    print("Mean: %.8f, Var: %.8f" % (np.mean(dataset[:, -2]), np.std(dataset[:, -2])))
    model = joblib.load(
        os.path.join(
            "model",
            "hpca07.mdl"
        )
    )
    pf = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    pred = np.exp(model.predict(pf.fit_transform(dataset[:, :-2])))[:, 0] * 10000
    # pred = model.predict(dataset[:, :-2])[:, 0] * 10000
    # plt.bar(range(len(dataset)), list(dataset[:, -2]), label='gt')
    # plt.bar(range(len(dataset)), list(pred), label='pred')
    plt.scatter(list(pred), dataset[:, -2], s=1.5, marker=markers[2])
    plt.xlabel("pred")
    plt.ylabel("gt")
    plt.title("HPCA07 on c.c")
    plt.grid()
    # plt.show()
    plt.savefig(
        os.path.join(
            "data",
            "figs",
            "HPCA07-pred.pdf"
        )
    )

if __name__ == "__main__":
    argv = parse_args()
    configs = get_configs(argv.configs)
    design_explorer_v3()
    # analysis()
