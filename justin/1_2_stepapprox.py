import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import models
from utils import NB_reparametrize, PSTH
import math
import scipy
import numpy as np

plt.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.titlesize': 18
})

'''
- to make close to ramp:

- E[x_t] for the diffusion process is given by E[x_t|not_absorbed]P(not_absorbed) + E[x_t|absorbed]P(absorbed)
- Try to approximate E[X_t] with the negative binomial CDF

- One method: P(absorbed) will be hard to compute so just approx E[x_t|not_absorbed] = E[x_t]
- Because the PSTH and negative binomial dists are discrete it could be possible to do this by least-squares or Gauss-Newton
'''
# default parameter guesses for solver
m0, r0 = 50, 10

# ramp parameters to fit to
beta = 0.5
sigma = 2
x0 = 0.2
T = 100

r0, p0 = NB_reparametrize(m0, r0)

# create the 'target' ramp PDF

def optimize_step(beta, sigma, x0, T, plot=False):
    # outputs the optimal step function to fit this ramp function
    # step function is parametrized as per m, r convention

    beta += 1e-8 # prevent div by zero

    avg_boundary_timestamp = math.floor(((1-x0) * T) / beta) # average TS where DD process hits
    if avg_boundary_timestamp < T:
        ramp_pdf = [beta/T] * avg_boundary_timestamp + [0] * (T-avg_boundary_timestamp)
    else:
        ramp_pdf = [beta/T] * T
    ramp_pdf = np.array(ramp_pdf)

    # least-squares optimize step paraxmeters w.r.t. ramp_pdf
    residuals = lambda theta: scipy.stats.nbinom.pmf(np.arange(0, T), *NB_reparametrize(*theta)) - ramp_pdf
    result = scipy.optimize.least_squares(residuals, x0=[r0, p0])

    if plot:
        plt.plot(ramp_pdf)
        plt.plot(scipy.stats.nbinom.pmf(np.arange(0, T), *NB_reparametrize(*result.x)))
        plt.show()

    return result.x

step_params = optimize_step(beta, sigma, x0, T, plot=True)
print(step_params)

# compare the PSTHs
step = models.StepModel(*step_params, x0=x0)
spikes, jumps, rates = step.simulate(Ntrials=10000, T=T)
counts, edges = PSTH(spikes, mode='valid')

plt.plot(edges[:-1], counts, label='step')


ramp = models.RampModel(beta=beta, sigma=sigma, x0=x0)
spikes, xs, rates = ramp.simulate(Ntrials=10000, T=T)
counts, edges = PSTH(spikes, mode='valid')


plt.plot(edges[:-1], counts, label='ramp')
plt.legend()
plt.show()

# seems to fail when: high sigma,high x0 - latter might be fixable by fitting CDFs
# fail when