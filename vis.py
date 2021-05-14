# Author: baichen318@gmail.com

import os
import matplotlib.pyplot as plt

markers = [
    '.', ',', 'o', 'v', '^', '<', '>', '1', '2', '3',
    '4', '8', 's', 'p', 'P', '*', 'h', 'H', '+', 'x',
    'X', 'D', 'd', '|', '_'
]
colors = [
    'c', 'b', 'g', 'r', 'm', 'y', 'k', 'w'
]

def plot_predictions_with_gt(gt, preds, **kwargs):
    """
        gt: <numpy.ndarray>
        preds: <numpy.ndarray>
    """
    plt.rcParams['savefig.dpi'] = 600
    plt.rcParams['figure.dpi'] = 600

    for i in range(len(gt)):
        p1 = plt.scatter(
            gt[i][0],
            gt[i][1],
            s=1.5,
            marker=markers[2],
            c=colors[-2]
        )
        p2 = plt.scatter(
            preds[i][0],
            preds[i][1],
            s=1.5,
            marker=markers[-8],
            c=colors[-3]
        )
    plt.legend(handles=[p1, p2], labels=["gt", "preds"])
    plt.xlabel('C.C.')
    plt.ylabel('Power')
    plt.title('C.C. vs. Power - ' + kwargs["title"])
    plt.grid()
    output = os.path.join(
        os.path.dirname(kwargs["output"]),
        kwargs["title"] + ".pdf"
    )
    print("[INFO]: save the figure", output)
    plt.savefig(output)
    plt.show()
