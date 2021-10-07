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
from util import parse_args, get_configs, load_txt, write_txt, mkdir
from dse.env.rocket.design_space import parse_design_space


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
    dataset = np.delete(dataset, remove_idx, axis=0)
    return dataset


def load_design_space():
    design_space = parse_design_space(
        configs["design-space"],
        basic_component=configs["basic-component"],
        random_state=configs["seed"],
    )
    return design_space


def split_dataset(dataset):
    # NOTICE: we omit the rest of groups
    kfold = KFold(n_splits=10, shuffle=True, random_state=configs["seed"])
    for train, test in kfold.split(dataset):
        return train, test


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


class PPADataset(BaseDataset):
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
        model.save(os.path.join(configs["ppa-model"], configs["design"] + '-' + metric + ".pt"))

def calib_mlp_test(design_space, dataset):
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
    L2 = nn.MSELoss()
    L1 = nn.L1Loss()
    lims = {
        "ipc": [0.630, 0.850],
    }
    for metric in metrics:
        print("[INFO]: test %s model." % metric)
        model = MLP(design_space.n_dim + 1, 1)
        model.load(os.path.join(configs["ppa-model"], configs["design"] + '-' + metric + ".pt"))
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
                l2)
            )
            plt.scatter(x, y, s=2, marker=markers[2], c=colors[1])
            plt.plot(
                np.linspace([lims[metric][0], lims[metric][1]], 1000),
                np.linspace([lims[metric][0], lims[metric][1]], 1000),
                c=colors[3],
                linewidth=1,
                ls='--'
            )
            plt.xlabel("Predicton")
            plt.ylabel("GT")
            plt.xlim(lims[metric])
            plt.ylim(lims[metric])
            plt.grid()
            plt.title("%s-%s" % (configs["design"], metric))
            plt.savefig(os.path.join("%s-%s.png" % (configs["design"], metric)))
            print("[INFO]: save figure to %s." % os.path.join("%s-%s.png" % (configs["design"], metric)))
            plt.close()


def main():
    dataset = load_dataset(os.path.join(os.path.pardir, configs["dataset-output-path"]))
    design_space = load_design_space()
    # construct pre-generated dataset
    new_dataset = []

    for data in dataset:
        print("[INFO]: evaluate microarchitecture:", data[:-3])
        ipc, power, area = design_space.evaluate_microarchitecture(
            configs,
            # architectural feature
            data[:-3].astype(int),
            1,
            split=True
        )
        new_dataset.append(
            np.insert(data, len(data), values=np.array([ipc, power, area * 1e6]), axis=0)
        )
        _new_dataset = np.array(new_dataset)
        write_txt(
            os.path.join(
                os.path.pardir,
                os.path.dirname(configs["dataset-output-path"]),
                os.path.splitext(os.path.basename(configs["dataset-output-path"]))[0] + "-E.txt"
            ),
            _new_dataset,
            fmt="%f"
        )
    dataset = load_dataset(os.path.join(
            os.path.pardir,
            os.path.dirname(configs["dataset-output-path"]),
            os.path.splitext(os.path.basename(configs["dataset-output-path"]))[0] + "-E.txt"
        )
    )
    train, test = split_dataset(dataset)
    train_data = torch.Tensor(dataset[train])
    test_data = torch.Tensor(dataset[test])
    calib_mlp_train(
        design_space,
        PPADataset(train_data, test_data, idx=design_space.n_dim)
    )
    calib_mlp_test(
        design_space,
        PPADataset(train_data, test_data, idx=design_space.n_dim)
    )


if __name__ == '__main__':
    configs = get_configs(parse_args().configs)
    metrics = ["ipc", "power", "area"]
    metrics = ["ipc"]
    configs["logger"] = None
    main()
