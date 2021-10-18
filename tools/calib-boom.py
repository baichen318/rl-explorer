# Author: baichen318@gmail.com

import sys, os
sys.path.insert(
    0,
    os.path.join(os.path.dirname(__file__), os.path.pardir)
)
sys.path.insert(
    0,
    os.path.join(os.path.dirname(__file__), os.path.pardir, "dse")
)
sys.path.insert(
    0,
    os.path.join(os.path.dirname(__file__), os.path.pardir, "util")
)
sys.path.insert(
    0,
    os.path.join(os.path.dirname(__file__), os.path.pardir, "vlsi")
)
import torch
import torch.nn as nn
import torch.utils.data as data
from abc import ABC, abstractmethod
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from scipy.stats import stats
try:
    from sklearn.externals import joblib
except ImportError:
    import joblib
from util import parse_args, get_configs, load_txt, write_txt, mkdir
from xgboost import XGBRegressor


markers = [
    '.', ',', 'o', 'v', '^', '<', '>', '1', '2', '3',
    '4', '8', 's', 'p', 'P', '*', 'h', 'H', '+', 'x',
    'X', 'D', 'd', '|', '_'
]
colors = [
    'c', 'b', 'g', 'r', 'm', 'y', 'k', # 'w'
]


def load_dataset(path):
    dataset = load_txt(path, fmt=float)
    remove_idx = []
    perf = dataset[:, -3]
    power = dataset[:, -2]
    area = dataset[:, -1]
    # remove invalid data
    for i in range(perf.shape[0]):
        if perf[i] == 0:
            remove_idx.append(i)
            continue
        if power[i] == 0:
            remove_idx.append(i)
            continue
        if area[i] == 0:
            remove_idx.append(i)
            continue
    # dataset = np.delete(dataset, remove_idx, axis=0)
    return dataset


def load_design_space():
    if configs["design"] == "rocket":
        from dse.env.rocket.design_space import parse_design_space
        design_space = parse_design_space(
            configs["design-space"],
            basic_component=configs["basic-component"],
            random_state=configs["seed"],
        )
    elif configs["design"] == "boom":
        from dse.env.boom.design_space import parse_design_space
        design_space = parse_design_space(
            configs["design-space"],
            basic_component=configs["basic-component"],
            random_state=configs["seed"],
        )
    else:
        assert configs["design"] == "cva6", \
            "[ERROR]: %s is not support." % configs["design"]
        exit(-1)
    return design_space


def split_dataset(dataset):
    # NOTICE: we omit the rest of groups
    kfold = KFold(n_splits=10, shuffle=True, random_state=configs["seed"])
    for train, test in kfold.split(dataset):
        return train, test


def init_xgb():
    """
        NOTICE: a checkpoint
    """

    return XGBRegressor(
        max_depth=3,
        n_estimators=1000,
        gamma=0.00001,
        min_child_weight=1,
        subsample=1,
        eta=0.3,
        reg_alpha=0.01,
        reg_lambda=0.01,
        booster="gbtree",
        objective="reg:squarederror",
        eval_metric="mae",
        n_jobs=-1
    )
    # return GridSearchCV(
    #     estimator=XGBRegressor(
    #         subsample=1.0,
    #         booster="gbtree",
    #         objective="reg:squarederror",
    #         n_jobs=-1
    #     ),
    #     param_grid={
    #         "max_depth": [i for i in range(3, 8)],
    #         "gamma": [0.0001, 0.00001, 0.000001, 0.0000001],
    #         "min_child_weight": [1, 2],
    #         "eta": [i for i in np.arange(0.05, 0.4, 0.01)],
    #         "reg_alpha": [i for i in np.arange(1.0, 3.0, 0.1)],
    #         "reg_lambda": [i for i in np.arange(0.1, 1, 0.01)],
    #     },
    #     cv=5
    # )

def init_lr():
    from sklearn.linear_model import LinearRegression
    return LinearRegression(n_jobs=-1)


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 100),
            nn.ReLU(),
            nn.Linear(100, 80),
            nn.ReLU(),
            nn.Linear(80, 60),
            nn.ReLU(),
            nn.Linear(60, output_dim)
        )

    def forward(self, x):
        return self.mlp(x)

    def save(self, path):
        torch.save(self.state_dict(), path)
        print("[INFO]: save model to %s." % path)

    def load(self, path):
        self.load_state_dict(torch.load(path))
        print("[INFO]: load model from %s." % path)


class BaseDataset(data.Dataset, ABC):
    def __init__(self):
        pass

    @abstractmethod
    def __len__(self):
        return 0

    @abstractmethod
    def __getitem__(self, index):
        pass


class PPADatasetV1(BaseDataset):
    def __init__(self, train, test, idx, batch=16):
        """
            train: <torch.Tensor>
            test: <torch.Tensor>
        """
        BaseDataset.__init__(self)
        self.train_ipc_feature = train[:, [i for i in range(idx)] + [-3]]
        self.train_ipc_gt = train[:, idx]
        self.train_power_feature = train[:, [i for i in range(idx)] + [-2]]
        self.train_power_gt = train[:, idx + 1]
        self.train_area_feature = train[:, [i for i in range(idx)] + [-1]]
        self.train_area_gt = train[:, idx + 2]
        self.test_ipc_feature = test[:, [i for i in range(idx)] + [-3]]
        self.test_ipc_gt = test[:, idx]
        self.test_power_feature = test[:, [i for i in range(idx)] + [-2]]
        self.test_power_gt = test[:, idx + 1]
        self.test_area_feature = test[:, [i for i in range(idx)] + [-1]]
        self.test_area_gt = test[:, idx + 2]
        self.batch = batch

        # remove zero elements from the test set
        remove_idx = []
        for i in range(self.test_ipc_gt.shape[0]):
            if np.equal(self.test_ipc_gt[i], 0):
                remove_idx.append(i)
            if np.equal(self.test_power_gt[i], 0):
                remove_idx.append(i)
            if np.equal(self.test_area_gt[i], 0):
                remove_idx.append(i)
        self.test_ipc_feature = np.delete(self.test_ipc_feature, remove_idx, axis=0)
        self.test_ipc_gt = np.delete(self.test_ipc_gt, remove_idx, axis=0)
        self.test_power_feature = np.delete(self.test_power_feature, remove_idx, axis=0)
        self.test_power_gt = np.delete(self.test_power_gt, remove_idx, axis=0)
        self.test_area_feature = np.delete(self.test_area_feature, remove_idx, axis=0)
        self.test_area_gt = np.delete(self.test_area_gt, remove_idx, axis=0)

    def __getitem__(self, index):
        return {
            "ipc": [
                self.train_ipc_feature[index],
                self.train_ipc_gt[index]
            ],
            "power": [
                self.train_power_feature[index],
                self.train_power_gt[index]
            ],
            "area": [
                self.train_area_feature[index],
                self.train_area_gt[index]
            ]
        }, {
            "ipc": [
                self.test_ipc_feature[index],
                self.test_ipc_gt[index]
            ],
            "power": [
                self.test_power_feature[index],
                self.test_power_gt[index]
            ],
            "area": [
                self.test_area_feature[index],
                self.test_area_gt[index]
            ]
        }

    def __len__(self):
        return len(self.train_ipc_feature)

    def get_test_data_size(self):
        return len(self.test_ipc_feature)


class PPADatasetV2(object):
    def __init__(self, train, test, idx):
        super(PPADatasetV2, self).__init__()
        self.train_ipc_feature = train[:, [i for i in range(idx)] + [-3]]
        self.train_ipc_gt = train[:, idx]
        self.train_power_feature = train[:, [i for i in range(idx)] + [-2]]
        self.train_power_gt = train[:, idx + 1]
        self.train_area_feature = train[:, [i for i in range(idx)] + [-1]]
        self.train_area_gt = train[:, idx + 2]
        self.test_ipc_feature = test[:, [i for i in range(idx)] + [-3]]
        self.test_ipc_gt = test[:, idx]
        self.test_power_feature = test[:, [i for i in range(idx)] + [-2]]
        self.test_power_gt = test[:, idx + 1]
        self.test_area_feature = test[:, [i for i in range(idx)] + [-1]]
        self.test_area_gt = test[:, idx + 2]

        # remove zero elements from the test set
        remove_idx = []
        for i in range(self.test_ipc_gt.shape[0]):
            if np.equal(self.test_ipc_gt[i], 0):
                remove_idx.append(i)
            if np.equal(self.test_power_gt[i], 0):
                remove_idx.append(i)
            if np.equal(self.test_area_gt[i], 0):
                remove_idx.append(i)
        self.test_ipc_feature = np.delete(self.test_ipc_feature, remove_idx, axis=0)
        self.test_ipc_gt = np.delete(self.test_ipc_gt, remove_idx, axis=0)
        self.test_power_feature = np.delete(self.test_power_feature, remove_idx, axis=0)
        self.test_power_gt = np.delete(self.test_power_gt, remove_idx, axis=0)
        self.test_area_feature = np.delete(self.test_area_feature, remove_idx, axis=0)
        self.test_area_gt = np.delete(self.test_area_gt, remove_idx, axis=0)


def calib_mlp_train(design_space, dataset):
    criterion = nn.MSELoss()
    for metric in metrics:
        print("[INFO]: train %s model." % metric)
        model = MLP(design_space.n_dim + 1, 1)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=configs[metric + "-model-lr"]
        )
        for epoch in range(configs["ppa-epoch"]):
            total_loss = 0
            for i, (train, _) in enumerate(dataset):
                pred = model(train[metric][0])
                loss = criterion(pred, train[metric][1])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            if (epoch + 1 ) % 100 == 0:
                print("[INFO]: Epoch[{}/{}], Loss: {:.8f}".format(
                        epoch + 1,
                        configs["ppa-epoch"],
                        loss.item()
                    )
                )
        mkdir(os.path.join(configs["ppa-model"]))
        model.save(os.path.join(configs["ppa-model"], configs["design"] + '-' + metric + "-torch.pt"))


def calib_mlp_test(design_space, dataset):
    plt.rcParams['savefig.dpi'] = 600
    plt.rcParams['figure.dpi'] = 600
    L2 = nn.MSELoss()
    L1 = nn.L1Loss()
    lims = {
        "rocket": {
            "ipc": [0.630, 0.850],
            "power": [0.002, 0.014],
            "area": [200000, 900000]
        },
        "boom": {
            "ipc": [0.620, 1.760],
            "power": [0.030, 0.14],
            "area": [1.20 * 1e6, 4.8 * 1e6]
        }
    }
    for metric in metrics:
        print("[INFO]: test %s model." % metric)
        model = MLP(design_space.n_dim + 1, 1)
        model.load(os.path.join(configs["ppa-model"], configs["design"] + '-' + metric + "-torch.pt"))
        model.eval()
        error, l1, l2 = 0, 0, 0
        x, y = [], []
        with torch.no_grad():
            for i, (_, test) in enumerate(dataset):
                pred = model(test[metric][0])
                x.append(float(pred))
                y.append(float(test[metric][1]))
                error += abs(float(pred) - float(test[metric][1])) / float(test[metric][1])
                _l1 = L1(pred, test[metric][1])
                _l2 = L2(pred, test[metric][1])
                l1 += _l1
                l2 += _l2
            print("[INFO]: {} model, error: {:.8f}, L1: {:.8f}, L2: {:.8f}".format(
                    metric,
                    error / dataset.get_test_data_size(),
                    l1,
                    l2
                )
            )
            plt.scatter(x, y, s=2, marker=markers[2], c=colors[1])
            plt.plot(
                np.linspace([lims[configs["design"]][metric][0], lims[configs["design"]][metric][1]], 1000),
                np.linspace([lims[configs["design"]][metric][0], lims[configs["design"]][metric][1]], 1000),
                c=colors[3],
                linewidth=1,
                ls='--'
            )
            plt.xlabel("Predicton")
            plt.ylabel("GT")
            plt.xlim(lims[configs["design"]][metric])
            plt.ylim(lims[configs["design"]][metric])
            plt.grid()
            plt.title("%s-%s" % (configs["design"], metric))
            plt.savefig(os.path.join("%s-%s.png" % (configs["design"], metric)))
            print("[INFO]: save figure to %s." % os.path.join("%s-%s.png" % (configs["design"], metric)))
            plt.close()


def calib_xgboost_train(dataset):
    for metric in metrics:
        print("[INFO]: train %s model." % metric)
        model = init_xgb()
        if metric == "ipc":
            model.fit(dataset.train_ipc_feature, dataset.train_ipc_gt)
        elif metric == "power":
            model.fit(dataset.train_power_feature, dataset.train_power_gt)
        else:
            assert metric == "area", "[ERROR]: unsupported metric."
            model.fit(dataset.train_area_feature, dataset.train_area_gt)
        if isinstance(model, GridSearchCV):
            print("[INFO]", type(model), model.best_estimator_, model.best_params_)
        mkdir(os.path.join(configs["ppa-model"]))
        joblib.dump(
            model,
            os.path.join(configs["ppa-model"], configs["design"] + '-' + metric + ".pt")
        )


def calib_xgboost_test(dataset):
    plt.rcParams['savefig.dpi'] = 600
    plt.rcParams['figure.dpi'] = 600
    lims = {
        "rocket": {
            "ipc": [0.630, 0.850],
            "power": [0.002, 0.014],
            "area": [200000, 900000]
        },
        "boom": {
            "ipc": [0.620, 1.760],
            "power": [0.030, 0.14],
            "area": [1.20 * 1e6, 4.8 * 1e6]
        }
    }
    for metric in metrics:
        print("[INFO]: test %s model." % metric)
        model = joblib.load(
            os.path.join(configs["ppa-model"], configs["design"] + '-' + metric + ".pt")
        )
        if metric == "ipc":
            pred = model.predict(dataset.test_ipc_feature)
            mae = mean_absolute_error(dataset.test_ipc_gt, pred)
            mse = mean_squared_error(dataset.test_ipc_gt, pred)
            error = np.mean((abs(pred - dataset.test_ipc_gt) / dataset.test_ipc_gt)) * 1e2
            kendall_tau, _ = stats.kendalltau(pred, dataset.test_ipc_gt)
            plt.scatter(pred, dataset.test_ipc_gt, s=2, marker=markers[2], c=colors[1])
        elif metric == "power":
            pred = model.predict(dataset.test_power_feature)
            mae = mean_absolute_error(dataset.test_power_gt, pred)
            mse = mean_squared_error(dataset.test_power_gt, pred)
            error = np.mean((abs(pred - dataset.test_power_gt) / dataset.test_power_gt)) * 1e2
            kendall_tau, _ = stats.kendalltau(pred, dataset.test_power_gt)
            plt.scatter(pred, dataset.test_power_gt, s=2, marker=markers[2], c=colors[1])
        else:
            assert metric == "area", "[ERROR]: unsupported metric."
            pred = model.predict(dataset.test_area_feature)
            mae = mean_absolute_error(dataset.test_area_gt, pred)
            mse = mean_squared_error(dataset.test_area_gt, pred)
            error = np.mean((abs(pred - dataset.test_area_gt) / dataset.test_area_gt)) * 1e2
            kendall_tau, _ = stats.kendalltau(pred, dataset.test_area_gt)
            plt.scatter(pred, dataset.test_area_gt, s=2, marker=markers[2], c=colors[1])
        print("[INFO] MAE: {:.8f}, MSE: {:.8f}, Error: {:.4f}%, Kendall Tau: {:.4f}".format(
                mae, mse, error, kendall_tau
            )
        )
        plt.plot(
            np.linspace(
                [lims[configs["design"]][metric][0], lims[configs["design"]][metric][1]],
                1000
            ),
            np.linspace(
                [lims[configs["design"]][metric][0], lims[configs["design"]][metric][1]],
                1000
            ),
            c=colors[3],
            linewidth=1,
            ls='--'
        )
        plt.xlabel("Predicton")
        plt.ylabel("GT")
        plt.xlim(lims[configs["design"]][metric])
        plt.ylim(lims[configs["design"]][metric])
        plt.grid()
        plt.title("{}-{} \n MAE: {:.8f} MSE: {:.8f} Error: {:.4f}% \n Kendall Tau: {:.4f}".format(
                configs["design"], metric, mae, mse, error, kendall_tau
            )
        )
        plt.savefig(os.path.join("%s-%s.png" % (configs["design"], metric)))
        print("[INFO]: save figure to %s." % os.path.join("%s-%s.png" % (configs["design"], metric)))
        plt.close()

def main():
    design_space = load_design_space()
    dataset = load_dataset(os.path.join(
            os.path.abspath(os.path.dirname(__file__)),
            os.path.pardir,
            os.path.dirname(configs["dataset-output-path"]),
            os.path.splitext(os.path.basename(configs["dataset-output-path"]))[0] + "-E.txt"
        )
    )
    train, test = split_dataset(dataset)
    if opt == "mlp":
        train_data = torch.Tensor(dataset[train])
        test_data = torch.Tensor(dataset[test])
        calib_mlp_train(
            design_space,
            PPADatasetV1(train_data, test_data, idx=design_space.n_dim)
        )
        calib_mlp_test(
            design_space,
            PPADatasetV1(train_data, test_data, idx=design_space.n_dim)
        )
    else:
        assert opt == "xgboost", "[ERROR]: unsupported method"
        calib_xgboost_train(
            PPADatasetV2(
                dataset[train],
                dataset[test],
                idx=design_space.n_dim
            )
        )
        calib_xgboost_test(
            PPADatasetV2(
                dataset[train],
                dataset[test],
                idx=design_space.n_dim
            )
        )


if __name__ == '__main__':
    configs = get_configs(parse_args().configs)
    metrics = ["ipc", "power", "area"]
    opt = "xgboost"
    configs["logger"] = None
    main()
