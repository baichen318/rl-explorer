# Author: baichen318@gmail.com

import os
from collections import OrderedDict
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from helper import get_configs, parse_args, read_csv, load_dataset, get_pareto_points


markers = [
    '.', ',', 'o', 'v', '^', '<', '>', '1', '2', '3',
    '4', '8', 's', 'p', 'P', '*', 'h', 'H', '+', 'x',
    'X', 'D', 'd', '|', '_'
]
colors = [
    'c', 'b', 'g', 'r', 'm', 'y', 'k', # 'w'
]

def plot_design_space(x, y):
    """
        dataset: <numpy.ndarray>
    """
    from handle_data import reference

    labels = ["Small", "Medium", "Large", "Mega", "Giga"]
    # plt.rcParams['savefig.dpi'] = 300
    # plt.rcParams['figure.dpi'] = 300

    for i in range(len(x)):
        plt.scatter(
            y[:, 0],
            y[:, 1],
            s=1,
            marker=markers[-6],
            c=colors[-2],
        )
    i = 0
    for d in reference:
        plt.scatter(
            d[0],
            d[1],
            s=1,
            marker=markers[-10],
            label=labels[i] + "BoomConfig"
        )
        i += 1
    plt.legend(loc="best", frameon=False)
    plt.xlabel('C.C.')
    plt.ylabel('Power')
    plt.title('C.C. vs. Power - ' + "design space")
    plt.grid()
    # output = os.path.join(
    #     os.path.dirname(configs["fig-output-path"]),
    #     "design-space" + ".pdf"
    # )
    # print("[INFO]: save the figure", output)
    plt.savefig("design-space.jpg")
    # plt.show()

def plot_design_space_wrt_clustering(x, y):
    plt.rcParams['savefig.dpi'] = 600
    plt.rcParams['figure.dpi'] = 600

    data = {}
    for i in range(len(x)):
        decode_width = str(int(x[i][1]))
        if str(decode_width) not in data.keys():
            data[decode_width] = []
        data[decode_width].append(y[i])

    cnt = 1
    for i in range(1, 6):
        for cc, power in data[str(i)]:
            plt.scatter(
                cc,
                power,
                s=1,
                marker=markers[i],
                c=colors[i]
            )
            if str(i) == "2":
                plt.annotate("%d" % cnt, (cc, power))
                cnt += 1
    plt.legend(loc="best", frameon=False)
    plt.xlabel("C.C.")
    plt.ylabel("Power")
    plt.title("C.C. vs. Power - " + "design space")
    plt.grid()
    plt.savefig("design-space.jpg")

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
        v = get_pareto_points(v[:, -2:])
        plt.scatter(
            v[:, 0],
            v[:, 1],
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

def plot_pareto_set(data, **kwargs):
    plt.rcParams['savefig.dpi'] = 600
    plt.rcParams['figure.dpi'] = 600

    dataset_path = kwargs["dataset_path"]
    x, y = load_dataset(dataset_path)

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(y[:, 0], y[:, 1], y[:, 2], s=1.5, marker=markers[19], c=colors[-1], alpha=0.2, label="GT")
    ax.scatter(data[:, 0], data[:, 1], data[:, 2], s=0.3,marker=markers[15],c=colors[3],label="Pareto set")
    ax.set_title("%BOOM PPA", pad=15, fontsize="10")
    ax.set_xlabel("IPC")
    ax.set_ylabel("Power")
    ax.set_zlabel("Area")
    ax.view_init(elev=3, azim=33)
    print("[INFO]: save the figure", kwargs["output"])
    plt.savefig(kwargs["output"])
    plt.show()

def main():
    x, y = load_dataset(configs["dataset-output-path"], preprocess=False)
    # plot_design_space(x, y)
    plot_design_space_wrt_clustering(x, y)

# def main(rpt=None):
#     data = OrderedDict()
#     for f in os.listdir(configs["rpt-output-path"]):
#         if rpt is not None:
#             if ".rpt" in f and rpt == f:
#                 data[f.split('.')[0]] = read_csv(
#                 os.path.join(
#                         configs["rpt-output-path"],
#                         f
#                     ),
#                     header=None
#                 )
#                 break
#         else:
#             if ".rpt" in f:
#                 data[f.split('.')[0]] = read_csv(
#                     os.path.join(
#                         configs["rpt-output-path"],
#                         f
#                     ),
#                     header=None
#                 )
#     plot_all_model_results(data)

if __name__ == "__main__":
    configs = get_configs(parse_args().configs)
    main()