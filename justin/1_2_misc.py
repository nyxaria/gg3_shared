import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import models
from utils import PSTH

plt.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.titlesize': 18
})

step = models.StepModel()
spikes, jumps, rates = step.simulate(Ntrials=5000, T=100)
counts, edges = PSTH(spikes, mode='valid')

plt.plot(edges[:-1], counts)
plt.show()


ramp = models.RampModel(beta=0.2, sigma=0.2, x0=0.2, Rh=50)
spikes, xs, rates = ramp.simulate(Ntrials=5000, T=100)
counts, edges = PSTH(spikes, mode='valid') # todo see if there's a better way, using bar is a bit unclean

plt.plot(edges[:-1], counts)
plt.show()