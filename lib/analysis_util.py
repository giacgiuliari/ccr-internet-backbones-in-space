import os
import pickle
import re

import matplotlib.pyplot as plt
import numpy as np


def scattermetric(data, metric, ax):
    nums = sorted(data.keys())

    for cur in nums:
        rero_points = data[cur]['rero'][metric]
        paco_points = data[cur]['paco'][metric]
        x_axis = np.zeros(len(paco_points)) + cur
        ax.scatter(x_axis, (rero_points / np.min(paco_points) - 1) * 100, c='c',
                   alpha=0.5, s=30)
        ax.scatter(x_axis, (paco_points / np.min(paco_points) - 1) * 100, c='m',
                   alpha=0.5, s=30)
        ax.set_ylabel(f"{metric} % loss from minimum achieved")
        ax.set_xlabel("# inactive GSTs")
    plt.tight_layout()


def load_all_runs(path_to_dir):
    data = {}
    all_hist_paco = []
    all_hist_rero = []
    all_percent_loss = []
    PACO_NAME = "pathcontrol_stat.pkl"
    RERO_NAME = "rerouting_stat.pkl"

    for cur_dir in os.listdir(path_to_dir):
        full_path = os.path.join(path_to_dir, cur_dir)
        if os.path.isfile(full_path):
            continue
        num_inactive = int(re.findall('\d+', cur_dir)[0])

        with open(os.path.join(full_path, PACO_NAME), 'rb') as infile:
            paco = pickle.load(infile)
        with open(os.path.join(full_path, RERO_NAME), 'rb') as infile:
            rero = pickle.load(infile)

        if num_inactive not in data:
            data[num_inactive] = {'paco': {}, 'rero': {}}
            data[num_inactive]['paco'] = {
                'max': [paco['cmax']],
                'min': [paco['cmin']],
                'avg': [paco['avg']],
                'q1': [paco['q1']],
                'q2': [paco['q2']],
                'q3': [paco['q3']],
                'hist': [paco['hist']],
            }
            all_hist_paco.append(paco['hist'][0])
            data[num_inactive]['rero'] = {
                'max': [rero['cmax']],
                'min': [rero['cmin']],
                'avg': [rero['avg']],
                'q1': [rero['q1']],
                'q2': [rero['q2']],
                'q3': [rero['q3']],
                'hist': [rero['hist']],
            }
            all_hist_rero.append(rero['hist'][0])
            all_percent_loss.append(rero['percent_loss'])

        else:
            data[num_inactive]['paco']['max'].append(paco['cmax'])
            data[num_inactive]['paco']['min'].append(paco['cmin'])
            data[num_inactive]['paco']['avg'].append(paco['avg'])
            data[num_inactive]['paco']['q1'].append(paco['q1'])
            data[num_inactive]['paco']['q2'].append(paco['q2'])
            data[num_inactive]['paco']['q3'].append(paco['q3'])
            data[num_inactive]['paco']['hist'].append(paco['hist'])
            all_hist_paco.append(paco['hist'][0])

            data[num_inactive]['rero']['max'].append(rero['cmax'])
            data[num_inactive]['rero']['min'].append(rero['cmin'])
            data[num_inactive]['rero']['avg'].append(rero['avg'])
            data[num_inactive]['rero']['q1'].append(rero['q1'])
            data[num_inactive]['rero']['q2'].append(rero['q2'])
            data[num_inactive]['rero']['q3'].append(rero['q3'])
            data[num_inactive]['rero']['hist'].append(rero['hist'])

            all_hist_rero.append(rero['hist'][0])
            all_percent_loss.append(rero['percent_loss'])

    return data, all_hist_paco, all_hist_rero, all_percent_loss


def plotmetric(metric, m_name, **kwargs):
    avg = np.average(metric, axis=0)
    maxim = np.max(metric, axis=0)
    minim = np.min(metric, axis=0)
    plt.plot(avg, label=f"Avg. {m_name}", **kwargs)
    plt.fill_between(np.arange(120), maxim, minim, alpha=0.2,
                     label=f"Range {m_name}")
