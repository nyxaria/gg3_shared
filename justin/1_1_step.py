import models
import numpy as np
import matplotlib.pyplot as plt
import math
import scipy

import torch
import torch.nn as nn
import torch.optim as optim
import sklearn

plt.style.use('ggplot')
# make text larger
plt.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.titlesize': 18
})

T = 1000
num_rasters = 3 # number of spike rasters per plot
num_params = 2 # number of params to test

fig, ax = plt.subplots(nrows=num_params, ncols=num_params, figsize=(5*num_params, 5*num_params))
# params to vary: m, r, x0, Rh

m = np.linspace(T * 0.25, T * 0.75, num_params)
r = np.linspace(3, 0, num_params)
r = np.pow(10, r)

x0 = 0.2
Rh = 50

'''m = T * 0.5
r = 5
x0 = [0.1, 0.5]
Rh = [25, 100]'''

for p1 in range(num_params): # to vary parameters todo
    for p2 in range(num_params):
        step = models.StepModel(m[p1], r[p2], x0, Rh)
        # step = models.StepModel(m, r, x0[p1], Rh[p2])

        spikes, jumps, rates = step.simulate(Ntrials=10, T=T)
        height = np.max(spikes)

        ax[p1, p2].set_title('m=' + str(m[p1]) + '; r=' + str(r[p2]))
        # ax[p1, p2].set_title('x0=' + str(x0[p1]) + '; Rh=' + str(Rh[p2]))

        for j in range(num_rasters):
            color = plt.rcParams['axes.prop_cycle'].by_key()['color'][j % len(plt.rcParams['axes.prop_cycle'])]

            ax[p1, p2].scatter(np.arange(len(spikes[j]))[spikes[j] > 0],
                               spikes[j][spikes[j] > 0] + (height * j), marker="s", color=color, s=7)

            ax[p1, p2].vlines(jumps[j], (height * j), (height * (j+1)), color=color)


            # binom mode
            ax[p1, p2].vlines(m[p1], (height * j), (height * (j+1)), color='black')

            ax[p1, p2].yaxis.set_tick_params(labelcolor='none')