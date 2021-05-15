# Author: baichen318@gmail.com

import os
import torch
import gpytorch
import numpy as np
from sample import ClusteringRandomizedTED
from dnn_gp import DNNGP, DNNGPV2
try:
    from sklearn.externals import joblib
except ImportError:
    import joblib
from sample import RandomSampler
from util import parse_args, get_configs, load_dataset, split_dataset, rmse, strflush
from vis import plot_predictions_with_gt
from space import parse_design_space
from exception import UnDefinedException

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
        {"D""params": model.gp.covar_module.parameters()},
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

# def design_explorer_v1():
#     dataset = get_data()
#     test_dataset, dataset = construct_test_dataset(dataset)
#     unsampled_dataset = dataset.copy()
#     sampled_dataset = []
#     sampler = ClusteringRandomizedTED(configs)
# 
#     # initialize
#     unsampled_dataset = sample(sampler, unsampled_dataset, sampled_dataset)
#     x, y = split_data(sampled_dataset)
#     # record R2
#     for i in range(configs["max-bo-steps"]):
#         model = fit_dnn_gp(x, y)
#         metrics = predict_by_dnn_gp(model, x, y)
#         msg = "[TRAIN] MSE (latency): %.8f, MSE (power): %.8f, " % (metrics[0], metrics[1]) + \
#             "MAPE (latency): %.8f, MAPE (power): %.8f, " % (metrics[4], metrics[5]) + \
#             "R2 (latency): %.8f, R2 (power): %.8f, " % (metrics[2], metrics[3]) + \
#             "training data size: %d" % len(sampled_dataset)
#         print(msg)
#         unsampled_dataset = sample(sampler, unsampled_dataset, sampled_dataset)
#         x, y = split_data(sampled_dataset)
# 
#         # evaluate
#         metrics = predict_by_dnn_gp(model, test_dataset[0], test_dataset[1])
#         msg = "[TEST] MSE (latency): %.8f, MSE (power): %.8f, " % (metrics[0], metrics[1]) + \
#                 "MAPE (latency): %.8f, MAPE (power): %.8f, " % (metrics[4], metrics[5]) + \
#                 "R2 (latency): %.8f, R2 (power): %.8f, " % (metrics[2], metrics[3]) + \
#                 "on test data"
#         print(msg)
# 
#     # validate on `unsampled_dataset`
#     x, y = split_data(unsampled_dataset)
#     metrics = predict_by_dnn_gp(model, x, y)
#     msg = "[ONE-MORE-TEST] MSE (latency): %.8f, MSE (power): %.8f, " % (metrics[0], metrics[1]) + \
#             "MAPE (latency): %.8f, MAPE (power): %.8f, " % (metrics[4], metrics[5]) + \
#             "R2 (latency): %.8f, R2 (power): %.8f, " % (metrics[2], metrics[3]) + \
#             "unsampled_dataset data size: %d" % len(unsampled_dataset)
#     print(msg)
#     # validate on `test_dataset`
#     metrics = predict_by_dnn_gp(model, test_dataset[0], test_dataset[1])
#     msg = "[FINAL-TEST] MSE (latency): %.8f, MSE (power): %.8f, " % (metrics[0], metrics[1]) + \
#             "MAPE (latency): %.8f, MAPE (power): %.8f, " % (metrics[4], metrics[5]) + \
#             "R2 (latency): %.8f, R2 (power): %.8f, " % (metrics[2], metrics[3]) + \
#             "on test data"
#     print(msg)
#     model.save(
#         os.path.join(
#             configs["model-output-path"],
#             "dnn-gp.mdl"
#         )
#     )

# def design_explorer_v2():
#     dataset = get_data()
#     test_dataset, dataset = construct_test_dataset(dataset)
#     unsampled_dataset = dataset.copy()
#     sampled_dataset = []
#     sampler = ClusteringRandomizedTED(configs)
# 
#     def analysis(gt, predict):
#         _mse = mse(gt, predict)
#         _r2 = r2(gt, predict)
#         _mape = mape(gt, predict)
# 
#         return _mse, _r2, _mape
# 
#     # initialize
#     unsampled_dataset = sample(sampler, unsampled_dataset, sampled_dataset)
#     x, y = split_data(sampled_dataset)
#     for i in range(configs["max-bo-steps"]):
#         model_l = DNNGPV2(configs, x, y[:, 0])
#         model_p = DNNGPV2(configs, x, y[:, 1])
#         model_l.fit(x, y[:, 0])
#         model_p.fit(x, y[:, 1])
# 
#         _y_l = model_l.predict(x)
#         _y_p = model_p.predict(x)
# 
#         metrics_l = analysis(y[:, 0], _y_l.cpu().detach().numpy())
#         metrics_p = analysis(y[:, 1], _y_p.cpu().detach().numpy())
# 
#         msg = "[TRAIN] MSE (latency): %.8f, MSE (power): %.8f, " % (metrics_l[0], metrics_p[0]) + \
#             "MAPE (latency): %.8f, MAPE (power): %.8f, " % (metrics_l[2], metrics_p[2]) + \
#             "R2 (latency): %.8f, R2 (power): %.8f, " % (metrics_l[1], metrics_p[1]) + \
#             "training data size: %d" % len(sampled_dataset)
#         print(msg)
#         unsampled_dataset = sample(sampler, unsampled_dataset, sampled_dataset)
#         x, y = split_data(sampled_dataset)
# 
#         # evaluate
#         _y_l = model_l.predict(test_dataset[0])
#         _y_p = model_p.predict(test_dataset[0])
# 
#         metrics_l = analysis(test_dataset[1][:, 0], _y_l.cpu().detach().numpy())
#         metrics_p = analysis(test_dataset[1][:, 1], _y_p.cpu().detach().numpy())
# 
#         msg = "[TEST] MSE (latency): %.8f, MSE (power): %.8f, " % (metrics_l[0], metrics_p[0]) + \
#             "MAPE (latency): %.8f, MAPE (power): %.8f, " % (metrics_l[2], metrics_p[2]) + \
#             "R2 (latency): %.8f, R2 (power): %.8f, " % (metrics_l[1], metrics_p[1]) + \
#             "on test data set"
#         print(msg)
# 
#     # validate on `unsampled_dataset`
#     x, y = split_data(unsampled_dataset)
#     _y_l = model_l.predict(x)
#     _y_p = model_p.predict(x)
# 
#     metrics_l = analysis(y[:, 0], _y_l.cpu().detach().numpy())
#     metrics_p = analysis(y[:, 1], _y_p.cpu().detach().numpy())
#     msg = "[FINAL-TEST] MSE (latency): %.8f, MSE (power): %.8f, " % (metrics_l[0], metrics_p[0]) + \
#         "MAPE (latency): %.8f, MAPE (power): %.8f, " % (metrics_l[2], metrics_p[2]) + \
#         "R2 (latency): %.8f, R2 (power): %.8f, " % (metrics_l[1], metrics_p[1]) + \
#         "unsampled_dataset data size: %d" % len(unsampled_dataset)
#     print(msg)
#     # model.save(
#     #     os.path.join(
#     #         configs["model-output-path"],
#     #         "dnn-gp.mdl"
#     #     )
#     # )

# def design_explorer_v3():
#     dataset = get_data()
#     test_dataset, dataset = construct_test_dataset(dataset)
# 
#     def analysis(gt, predict):
#         _mse = mse(gt, predict)
#         _r2 = r2(gt, predict)
#         _mape = mape(gt, predict)
# 
#         return _mse, _r2, _mape
# 
#     # initialize
#     x, y = split_data(dataset)
#     model_l = DNNGPV2(configs, x, y[:, 0])
#     model_p = DNNGPV2(configs, x, y[:, 1])
#     model_l.fit(x, y[:, 0])
#     model_p.fit(x, y[:, 1])
# 
#     _y_l = model_l.predict(x)
#     _y_p = model_p.predict(x)
# 
#     metrics_l = analysis(y[:, 0], _y_l.cpu().detach().numpy())
#     metrics_p = analysis(y[:, 1], _y_p.cpu().detach().numpy())
# 
#     msg = "[TRAIN] MSE (latency): %.8f, MSE (power): %.8f, " % (metrics_l[0], metrics_p[0]) + \
#         "MAPE (latency): %.8f, MAPE (power): %.8f, " % (metrics_l[2], metrics_p[2]) + \
#         "R2 (latency): %.8f, R2 (power): %.8f, " % (metrics_l[1], metrics_p[1]) + \
#         "training data size: %d" % len(x)
#     print(msg)
# 
#     # evaluate
#     _y_l = model_l.predict(test_dataset[0])
#     _y_p = model_p.predict(test_dataset[0])
# 
#     metrics_l = analysis(test_dataset[1][:, 0], _y_l.cpu().detach().numpy())
#     metrics_p = analysis(test_dataset[1][:, 1], _y_p.cpu().detach().numpy())
# 
#     msg = "[TEST] MSE (latency): %.8f, MSE (power): %.8f, " % (metrics_l[0], metrics_p[0]) + \
#         "MAPE (latency): %.8f, MAPE (power): %.8f, " % (metrics_l[2], metrics_p[2]) + \
#         "R2 (latency): %.8f, R2 (power): %.8f, " % (metrics_l[1], metrics_p[1]) + \
#         "on test data set"
#     print(msg)
# 
#     model_l.save(
#         os.path.join(
#             configs["model-output-path"],
#             "dnn-gp-cc.mdl"
#         )
#     )
#     model_p.save(
#         os.path.join(
#             configs["model-output-path"],
#             "dnn-gp-power.mdl"
#         )
#     )

class SurrogateModel(object):
    """
        SurrogateModel: 12 traditional basic ML models
    """
    def __init__(self, configs, train_x, train_y):
        super(SurrogateModel, self).__init__()
        self.model = DNNGPV2(configs, torch.Tensor(train_x), torch.Tensor(train_y))

    def fit(self, x, y):
        """
            x: <numpy.ndarray>
            y: <numpy.ndarray>
        """
        self.model.fit(torch.Tensor(x), torch.Tensor(y))

    def predict(self, x):
        """
            x: <numpy.ndarray>
        """
        return self.model.predict(torch.Tensor(x))

    def save(self, path):
        self.model.save(path)

    def load(self, path):
        self.model.load(path)

class BayesianOptimization(object):
    """docstring for BayesianOptimization"""
    def __init__(self, configs):
        super(BayesianOptimization, self).__init__()
        self.configs = configs
        # build model
        self.model1 = None
        self.model2 = None
        self.space = parse_design_space(self.configs["design-space"])
        self.sampler = RandomSampler(self.configs)
        self.dataset = load_dataset(configs["dataset-output-path"])
        self.unsampled = None

    def sample(self, dataset):
        return self.sampler.sample(dataset)

    # def fit(self, x, y):
    #     self.model.fit(x, y)

    # def predict(self, x):
    #     return self.model.predict(x)

    def run(self):
        x, y = [], []
        dataset = self.dataset.copy()
        for i in range(self.configs["max-bo-steps"]):
            dataset, sample = self.sample(dataset)
            _x, _y = split_dataset(sample)
            # add `_x` & `_y` to `x` & `y` respectively
            if len(x) == 0:
                for j in _x:
                    x.append(j)
                x = np.array(x)
                for j in _y:
                    y.append(j)
                y = np.array(y)
            else:
                for j in _x:
                    x = np.insert(x, len(x), j, axis=0)
                for j in _y:
                    y = np.insert(y, len(y), j, axis=0)
            self.model1 = SurrogateModel(configs, x, y[:, 0])
            self.model2 = SurrogateModel(configs, x, y[:, 1])
            self.model1.fit(x, y[:, 0])
            self.model2.fit(x, y[:, 1])

            y1 = self.model1.predict(x).numpy()
            y2 = self.model2.predict(x).numpy()
            # print("train:", y1)
            # print("train:", y2)
            # print("train:", y)
            msg = "[INFO]: Training Iter %d: RMSE of c.c.: %.8f, " % ((i + 1), rmse(y[:, 0], y1)) + \
                "RMSE of power: %.8f on %d train data" % (rmse(y[:, 1], y2), len(x))
            strflush(msg)
            # validate
            _x, _y = split_dataset(dataset)
            y1 = self.model1.predict(_x).numpy()
            y2 = self.model2.predict(_x).numpy()
            # print("test:", y1)
            # print("test:", y2)
            # print("test:", _y)
            msg = "[INFO]: Testing Iter %d: RMSE of c.c.: %.8f, " % ((i + 1), rmse(_y[:, 0], y1)) + \
                "RMSE of power: %.8f on %d test data" % (rmse(_y[:, 1], y2), len(_x))
            strflush(msg)

        self.unsampled = dataset

    def validate(self, logger=None):
        x, y = split_dataset(self.unsampled)
        y1 = self.model1.predict(x).numpy()
        y2 = self.model2.predict(x).numpy()
        msg = "[INFO]: RMSE of c.c.: %.8f, " % rmse(y[:, 0], y1) + \
            "RMSE of power: %.8f on %d test data" % (rmse(y[:, 1], y2), len(self.unsampled))
        strflush(msg)
        _y = []
        for i in range(len(y1)):
            _y.append([y1[i], y2[i]])
        _y = np.array(_y)
        # visualize
        plot_predictions_with_gt(
            y,
            _y,
            title="dnn-gp",
            output=self.configs["fig-output-path"]
        )

    def save(self):
        output = os.path.join(
            self.configs["model-output-path"],
            "dnn-gp-1"+ ".mdl"
        )
        self.model1.save(output)
        output = os.path.join(
            self.configs["model-output-path"],
            "dnn-gp-2"+ ".mdl"
        )
        self.model2.save(output)

    def load(self):
        output = os.path.join(
            self.configs["model-output-path"],
            "dnn-gp-1"+ ".mdl"
        )
        self.model1.load(output)
        output = os.path.join(
            self.configs["model-output-path"],
            "dnn-gp-2"+ ".mdl"
        )
        self.model2.load(output)

def main():
    manager = BayesianOptimization(configs)
    manager.run()
    manager.validate()
    manager.save()

if __name__ == "__main__":
    configs = get_configs(parse_args().configs)
    main()

