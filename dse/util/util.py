# Author: baichen318@gmail.com

import os
import csv
import numpy as np

def if_exist(path, strict=False):
    try:
        if os.path.exists(path):
            return True
        else:
            raise NotFoundException(path)
    except NotFoundException as e:
        print(e)
        if not strict:
            return False
        else:
            exit(1)

def load_dataset(csv_path, preprocess=True):
    """
        csv_path: <str>
    """
    def _read_csv(csv_path):
        data = []
        if_exist(csv_path, strict=True)
        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            title = next(reader)
            for row in reader:
                data.append(row)

        return data, title

    def validate(dataset):
        """
            `dataset`: <tuple>
        """
        data = []
        for item in dataset:
            _data = []
            f = item[0].split(' ')
            for i in f:
                _data.append(int(i))
            for i in item[1:]:
                _data.append(float(i))
            data.append(_data)
        data = np.array(data)

        return data

    dataset, _ = _read_csv(csv_path)
    dataset = validate(dataset)
    x = []
    y = []
    for data in dataset:
        x.append(data[:-2])
        if preprocess:
            # NOTICE: scale the data by `max - x / \alpha`
            # The larger the value, the better the performance is
            y.append(np.array([(90000 - data[-2]) / 20000, (0.2 - data[-1]) * 10]))
        else:
            y.append(np.array([data[-2], data[-1]]))

    return np.array(x), np.array(y)
