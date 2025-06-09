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

num_rasters=3
num_param1=2
num_param2=2
fig, ax = plt.subplots(nrows=num_param1, ncols=num_param2, figsize=(5*num_param1, 5*num_param2))
fig2, ax_rate = plt.subplots(nrows=num_param1, ncols=num_param2, figsize=(5*num_param1, 5*num_param2))

beta = np.linspace(0.2, 2, num_param1)
sigma = np.linspace(0.1, 1, num_param2)

for p1 in range(num_param1):
  for p2 in range(num_param2):
    ramp = models.RampModel(beta[p1], sigma[p2])
    spikes_ramp, xs, rates_ramp = ramp.simulate(Ntrials=10, T=1000)

    height_ramp = np.max(spikes_ramp)
    # print(height_ramp)
    # height_ramp = 3 # this is horrible code

    ax[p1, p2].set_title("beta="+str(np.round(beta[p1], 1)) + "; sigma="+str(np.round(sigma[p2], 1)))
    ax_rate[p1, p2].set_title("beta="+str(np.round(beta[p1], 1)) + "; sigma="+str(np.round(sigma[p2], 1)))
    for j in range(num_rasters):
      color = plt.rcParams['axes.prop_cycle'].by_key()['color'][j % len(plt.rcParams['axes.prop_cycle'])]

      ax[p1, p2].scatter(np.arange(len(spikes_ramp[j]))[spikes_ramp[j] > 0],
                               spikes_ramp[j][spikes_ramp[j] > 0] + (height_ramp * j), marker="s", color=color, s=7)
      ax_rate[p1, p2].plot(np.arange(len(xs[j])),xs[j])

      threshold = np.argmax(xs[j]==1)
      if not threshold == 0: # no threshold
          ax[p1, p2].vlines(threshold, (height_ramp * j), (height_ramp * (j+1)), color=color)
          ax_rate[p1, p2].vlines(threshold, 0, 1, color=color)

      ax[p1, p2].yaxis.set_tick_params(labelcolor='none')