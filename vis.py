# Author: baichen318@gmail.com

import os
from collections import OrderedDict
import matplotlib.pyplot as plt
from util import get_configs, parse_args, read_csv, load_dataset

markers = [
    '.', ',', 'o', 'v', '^', '<', '>', '1', '2', '3',
    '4', '8', 's', 'p', 'P', '*', 'h', 'H', '+', 'x',
    'X', 'D', 'd', '|', '_'
]
colors = [
    'c', 'b', 'g', 'r', 'm', 'y', 'k', # 'w'
]

def plot_predictions_with_gt(gt, preds, highlight, **kwargs):
    """
        gt: <numpy.ndarray>
        preds: <numpy.ndarray>
    """
    plt.rcParams['savefig.dpi'] = 600
    plt.rcParams['figure.dpi'] = 600

    handler = []
    labels = []
    for i in range(len(gt)):
        plt.scatter(
            gt[i][0],
            gt[i][1],
            s=2,
            marker=markers[2],
            c=colors[-2],
            label="GT"
        )
    # transform layout
    x, y = [], []
    for i in range(kwargs["top_k"]):
        _x, _y = [], []
        _x.append(preds[i][0])
        _x.append(highlight[i][0])
        _y.append(preds[i][1])
        _y.append(highlight[i][1])
        x.append(_x)
        y.append(_y)
        labels.append("Top %s" % str(i + 1))

    for i in range(kwargs["top_k"]):
        handler.append(
            plt.scatter(
                x[i],
                y[i],
                s=1,
                marker=markers[i % len(markers)],
                c=colors[i % len(colors)],
                label="Top %s" % str(i + 1)
            )
        )
    plt.legend(handles=handler, labels=labels)
    plt.xlabel('C.C.')
    plt.ylabel('Power')
    plt.title('C.C. vs. Power - ' + kwargs["title"])
    # plt.grid()
    output = os.path.join(
        os.path.dirname(kwargs["output"]),
        kwargs["title"] + ".pdf"
    )
    print("[INFO]: save the figure", output)
    plt.savefig(output)
    # plt.show()

def plot_all_model_results(data):
    """
        data: <dict>
    """
    plt.rcParams['savefig.dpi'] = 600
    plt.rcParams['figure.dpi'] = 600
    dataset = load_dataset(configs["dataset-output-path"])
    dataset[:, -2] = dataset[:, -2] / 90000
    dataset[:, -1] = dataset[:, -1] * 10

    i = 0
    for k, v in data.items():
        plt.scatter(
            v[0, -2],
            v[0, -1],
            s=1.5,
            marker=markers[i % len(markers)],
            c=colors[i % len(colors)],
            label=k
        )
        i += 1
    plt.scatter(
        dataset[:, -2],
        dataset[:, -1],
        s=0.2,
        marker=markers[19],
        c=colors[-1],
        alpha=0.2
    )
    plt.legend(loc="best", ncol=3, fontsize=3)
    plt.xlabel('C.C.')
    plt.ylabel('Power')
    plt.title('C.C. vs. Power Model Comparison')
    output = os.path.join(
        os.path.dirname(configs["fig-output-path"]),
        "final-result" + ".pdf"
    )
    print("[INFO]: save the figure", output)
    plt.savefig(output)
    plt.show()

def main():
    data = OrderedDict()
    for f in os.listdir("rpts"):
        if ".rpt" in f:
            data[f.split('.')[0]] = read_csv(
                os.path.join(
                    "rpts",
                    f
                ),
                header=None
            )
    plot_all_model_results(data)

if __name__ == "__main__":
    configs = get_configs(parse_args().configs)
    main()
