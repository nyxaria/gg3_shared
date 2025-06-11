import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.titlesize': 18
})

import math
import scipy

import torch
import torch.nn as nn
import torch.optim as optim
import sklearn



def NB_reparametrize(p1, p2, reverse=False): # helper function to convert NB parameters
    # normal converts (μ, r) -> (r, p)
    # reverse converts (r, p) -> (μ, r)
    if reverse:
        return p1*(1-p2)/p2, p1
    else:
        return p2, p2/(p2+p1)


def PSTH(spikes, filter=np.tile(1/5, 5), mode='same', bins=None): # default: MA of 5 TS
    nz = np.nonzero(spikes)
    time_indices = np.repeat(nz[1], spikes[nz])
    if bins is None:
        bins = np.arange(np.min(time_indices), np.max(time_indices)+2)
    counts, edges = np.histogram(time_indices, bins=bins, density=True) # tuple (frequencies, bins)

    if mode=='valid':
        edges = edges[:-len(filter)+1] if len(filter) > 1 else edges
    # full not implemented
    return (np.convolve(counts, filter, mode=mode), edges)

def symmetric_kl(p, q):
    max_len = max(len(p), len(q))

    p_padded = np.pad(p, (0, max_len - len(p)), 'constant')
    q_padded = np.pad(q, (0, max_len - len(q)), 'constant')
    skl = scipy.stats.entropy(p_padded, q_padded) + scipy.stats.entropy(p_padded, q_padded)
    return skl

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

def compute_isi_distribution(spike_trains, count_multiple_spikes = False):
    assert not count_multiple_spikes # too bad
    all_isis = []

    for spike_train in spike_trains:
        # Get spike positions and counts (handles multi-spike bins)
        spike_positions = np.where(spike_train > 0)[0]  # Time bins with >=1 spike
        spike_counts = spike_train[spike_positions]     # Number of spikes in each bin

        # Generate fractional spike times for multi-spike bins
        expanded_spike_times = [-1]
        for pos, count in zip(spike_positions, spike_counts):
            if count > 1:
                if not count_multiple_spikes: # just set to pos, 1, 1, ...
                    # print(pos, count)
                    a = [pos] * count
                    # print(a)
                    expanded_spike_times += a
            else:
                expanded_spike_times.append(pos)

        expanded_spike_times = np.array(expanded_spike_times)

        isis = np.diff(expanded_spike_times)

        '''# Compute ISIs (differences between consecutive spikes)
        if len(expanded_spike_times) >= 2:
            isis = np.diff(expanded_spike_times)
        else:
            isis = np.array([])  # No ISIs if fewer than 2 spikes'''

        all_isis.append(isis)


    all_isis = np.concatenate(all_isis)
    hist = np.histogram(all_isis.flatten(), bins=np.arange(0, spike_trains.shape[1]+1), density=True)

    # print(spike_trains)
    # print(hist)
    return hist

def compute_summary_statistics(spike_trains, batch_size, bin_width=5):
    s = spike_trains.shape
    # print(s)

    # X = np.zeros((s[0] // batch_size, (s[1] // bin_width) * 3 + 1)) # I don't even know anymore
    X = []
    # print('before', spike_trains[0], spike_trains.shape)
    spike_trains_binned = spike_trains[:, 0:(s[1]//bin_width) * bin_width].reshape(s[0], -1, bin_width).sum(2)
    # print('after', spike_trains_binned[0], spike_trains_binned.shape)
    spike_trains = spike_trains_binned # lazy

    for first in range(0, spike_trains.shape[0], batch_size):
        batch = spike_trains[first:first+batch_size]

        #print(batch)

        if not np.any(batch):
            print('batch was all zeros')

        # smoothed_batch = scipy.ndimage.gaussian_filter1d(batch, sigma=1, output=np.float64)

        fano = np.var(batch, axis=0) / (np.mean(batch, axis=0) + 1e-8) # implicitly use every bin
        # consider  + 1e-8 to avoid div by 0
        psth, _ = PSTH(batch, bins=np.arange(spike_trains.shape[1]+1))
        isi, _ = compute_isi_distribution(batch)

        X.append(np.concatenate((fano, psth, isi)))
        # X[first//batch_size] = np.concatenate((fano, psth, isi))
    X = np.array(X)
    # print(X.shape)
    return X


class StepRampClassifier(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(input_dim, 15),
            nn.ReLU(),
            nn.Linear(15, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.layers(x)
