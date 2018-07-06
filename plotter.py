import argparse
import tensorflow as tf
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from tensorboard.backend.event_processing.event_multiplexer import EventMultiplexer

from matplotlib import pyplot as plt
from scipy.interpolate import spline
import numpy as np
from matplotlib import colors as colors
import seaborn as sns

plt.switch_backend('agg')
sns.set(style="darkgrid")
sns.set_context("paper")


def plot(params):
    ''' beautify tf log
        Use better library (seaborn) to plot tf event file'''

    log_path = params['logdir']
    fname = params['filename']
    title = params['title']

    accs = EventMultiplexer().AddRunsFromDirectory(log_path)
    accs.Reload()

    scalars = []
    name = []
    order = []
    for a in accs._accumulators:
        acc = accs._accumulators[a]
        s = acc.Scalars('val_acc')
        scalars.append(s)
        clu = a.split('_')[3]
        wind = a.split('_')[1]
        order.append([clu, wind])
        n = '_'.join(a.split('_')[:-1])
        name.append(n)

    scalars = [x for _, x in sorted(sorted(zip(order, scalars), key=lambda x: x[0][0]), key=lambda x:x[0][1])]
    name = [x for _, x in sorted(sorted(zip(order, name), key=lambda x: x[0][0]), key=lambda x:x[0][1])]

    x_list = []
    y_list = []
    x_list_raw = []
    y_list_raw = []
    for scalar in scalars:
        x = [int(s.step) for s in scalar]
        y = [s.value for s in scalar]

        # smooth curve
        x_sm = np.array(x)
        y_sm = np.array(y)
        x_ = np.linspace(x_sm.min(), x_sm.max(), 200)
        y_ = spline(x, y, x_)
        x_list.append(x_)
        y_list.append(y_)

    figure = plt.figure(figsize=(16,8))
    ax = plt.subplot(111)

    for i in range(len(x_list)):
        plt.title(title)
        plt.plot(x_list[i], y_list[i])
    ax.legend(labels=name, bbox_to_anchor=(1.12, 1), frameon=True, edgecolor='black', title = 'Legend')
    plt.savefig(fname=fname+"plot.png")
    plt.close(figure)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', default='./logs_r/wiki/', type=str, help='logdir to event file')
    parser.add_argument('--filename', default="logs_r/wiki/", help='name of the file for the plot')
    parser.add_argument('--title', default="Wikipedia Accuracy", help='title of the plot' )

    args = parser.parse_args()
    params = vars(args)  # convert to ordinary dict

    plot(params)
