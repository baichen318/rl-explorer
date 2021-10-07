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
from sklearn.model_selection import KFold
from util import parse_args, get_configs, load_txt, write_txt
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
    return KFold(n_splits=10, shuffle=True, random_state=configs["seed"])


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 1000),
            nn.ReLU(),
            nn.Linear(1000, 500),
            nn.ReLU(),
            nn.Linear(500, 50),
            nn.ReLU(),
            nn.Linear(50, output_dim)
        )

    def forward(self, x):
        return self.mlp(x)


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
    def __init__(self, train, test, idx):
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

def calib_train(design_space, dataset, num_epochs=10):
    criterion = nn.MSELoss()
    for metric in ["ipc", "power", "area"]:
        model = MLP(design_space.n_dim + 1, 1)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        for epoch in range(num_epochs):
            for i, (train, test) in enumerate(dataset):
                pred = model(train[metric][0])
                loss = criterion(pred, train[metric][1])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if (i + 1) % 50 == 0:
                    print("[INFO]: Epoch[{}/{}], step[{}/{}]".format(
                            epoch + 1,
                            num_epochs + 1,
                            i + 1,
                            len(dataset),
                            loss.item()
                        )
                    )


def main():
    dataset = load_dataset(os.path.join(os.path.pardir, configs["dataset-output-path"]))
    design_space = load_design_space()
    # construct pre-generated dataset
    new_dataset = []
    # for data in dataset:
    #     print("[INFO]: evaluate microarchitecture:", data[:-3])
    #     ipc, power, area = design_space.evaluate_microarchitecture(
    #         configs,
    #         # architectural feature
    #         data[:-3].astype(int),
    #         1,
    #         split=True
    #     )
    #     new_dataset.append(
    #         np.insert(data, len(data), values=np.array([ipc, power, area * 1e6]), axis=0)
    #     )
    #     _new_dataset = np.array(new_dataset)
    #     write_txt(
    #         os.path.join(
    #             os.path.pardir,
    #             os.path.dirname(configs["dataset-output-path"]),
    #             os.path.splitext(os.path.basename(configs["dataset-output-path"]))[0] + "-E.txt"
    #         ),
    #         _new_dataset,
    #         fmt="%f"
    #     )
    dataset = load_dataset(os.path.join(
            os.path.pardir,
            os.path.dirname(configs["dataset-output-path"]),
            os.path.splitext(os.path.basename(configs["dataset-output-path"]))[0] + "-E.txt"
        )
    )
    print("[TEST]:", dataset, dataset.shape)
    kfold = split_dataset(dataset)
    for train, test in kfold.split(dataset):
        train_data = torch.Tensor(dataset[train])
        test_data = torch.Tensor(dataset[test])
        calib_train(
            design_space,
            PPADataset(train_data, test_data, idx=design_space.n_dim)
        )






if __name__ == '__main__':
    configs = get_configs(parse_args().configs)
    configs["logger"] = None
    main()
