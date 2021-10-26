# baichen318@gmail.com

import sys, os
sys.path.insert(
    0,
    os.path.join(os.path.dirname(__file__), os.path.pardir)
)
import torch
import gpytorch
import tqdm
import numpy as np
import matplotlib.pyplot as plt
from boom_design_problem import BOOMDesignProblem
from dnn_gp import DNNGP, DNNGPV2
from sample import random_sample_v2
from util import parse_args, get_configs

def init_model():
    from sklearn.linear_model import LinearRegression
    return LinearRegression(n_jobs=-1)

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

def main():
    # ratio = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    # ratio = [0.1]
    problem = BOOMDesignProblem(configs)
    latency_mae, power_mae, latency_mse, power_mse, latency_err, power_err = [], [], [], [], [], []
    train_x, train_y = torch.Tensor([]), torch.Tensor([])
    ratio = []
    b, cnt = 0.5, 1
    num = int(round(b * problem.n_sample))
    while cnt <= 8:
        x, y = random_sample_v2(configs, problem, num)
        train_x, train_y = torch.cat((train_x, x)), torch.cat((train_y, y))
        dkl_gp = fit_dnn_gp(train_x, train_y)
        pred = dkl_gp.predict(problem.x.float())
        latency_mae.append(torch.nn.L1Loss()(pred[:, 0], problem.y[:, 0]))
        power_mae.append(torch.nn.L1Loss()(pred[:, 1], problem.y[:, 1]))
        latency_err.append(torch.mean(torch.abs(pred[:, 0] - problem.y[:, 0]) / problem.y[:, 0]))
        power_err.append(torch.mean(torch.abs(pred[:, 1] - problem.y[:, 1]) / problem.y[:, 1]))
        latency_mse.append(torch.nn.MSELoss()(pred[:, 0], problem.y[:, 0]))
        power_mse.append(torch.nn.MSELoss()(pred[:, 1], problem.y[:, 1]))
        # lr = init_model()
        # lr.fit(train_x.numpy(), train_y[:, 0].numpy())
        # pred = lr.predict(problem.x.numpy())
        # print(pred.shape.expand_dims())
        # # latency_err.append()
        # exit()
        ratio.append(b * cnt)
        cnt += 1
        break
        # print(latency_err, power_err, latency_mae, power_mae, latency_mse, power_mse)
    for idx in range(len(ratio)):
        print("ratio: %f, Latency MAE: %f, Power MAE: %f, Latency MSE: %f, Power MSE: %f, Latency err: %f, Power err: %f" %
            (ratio[idx], latency_mae[idx], power_mae[idx], latency_mse[idx], power_mse[idx], latency_err[idx], power_err[idx])
        )


    # plot
    markers = [
        '.', ',', 'o', 'v', '^', '<', '>', '1', '2', '3',
        '4', '8', 's', 'p', 'P', '*', 'h', 'H', '+', 'x',
        'X', 'D', 'd', '|', '_'
    ]
    colors = [
        'c', 'b', 'g', 'r', 'm', 'y', 'k', # 'w'
    ]

    plt.rcParams['savefig.dpi'] = 600
    plt.rcParams['figure.dpi'] = 600
    plt.plot(ratio, latency_err, 'r--', label="Err. of C.C.")
    plt.plot(ratio, power_err, 'b-', label="Err. of Power")
    plt.xlabel("Sample Ratio")
    plt.ylabel("Err.")
    # plt.ylim((0, 1))
    plt.title("Metrics v.s. Sample Ratio")
    plt.legend()
    plt.savefig("fig.png")
    plt.show()
    plt.close()



if __name__ == "__main__":
    configs = get_configs(parse_args().configs)
    main()
