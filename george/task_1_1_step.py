import numpy as np
import matplotlib.pyplot as plt
import models
import argparse
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

parser = argparse.ArgumentParser()
parser.add_argument('--show', action='store_true')
args = parser.parse_args()

N_TRIALS = 50  # number of trials to simulate
T_DURATION = 1000  # duration of each trial in time-steps or ms

M = T_DURATION / 2  # mean jump time
R_values = [10, 100, 1000, 10000] # jump time variability 

for R in R_values:
    print(f"\nTesting R = {R}")
    step_model = models.StepModel(m=M, r=R)

    spikes_step, jumps_step, rates_step = step_model.simulate(Ntrials=N_TRIALS, T=T_DURATION, get_rate=True)

    # spike raster
    plt.figure(figsize=(10, 6))
    for trial_idx in range(N_TRIALS):
        spike_times_trial = np.where(spikes_step[trial_idx, :] > 0)[0]
        plt.plot(spike_times_trial, np.ones_like(spike_times_trial) * trial_idx, '|', color='black', markersize=5)
        
        plt.plot(jumps_step[trial_idx], trial_idx, 'ro', markersize=5, label='Jump Time' if trial_idx == 0 else "")

    plt.xlabel("Time (ms)")
    plt.ylabel("Trial Number")
    plt.title(f"Step Model - Spike Raster (m={step_model.m}, r={R})")
    plt.ylim(-1, N_TRIALS)
    if N_TRIALS > 0: plt.legend()
    plt.savefig(f'plots/task_1_1_step_raster_R{R}.png', dpi=300, bbox_inches='tight')
    if args.show:
        plt.show()
    else:
        plt.close()

    # histogram
    plt.figure(figsize=(8, 5))
    plt.hist(jumps_step, bins=50, range=(0, 1000), density=True, color='skyblue', edgecolor='black')
    plt.xlabel("Time (ms)")
    plt.ylabel("Frequency (Hz)")
    plt.title(f"Step Model - Histogram of Jump Times (m={step_model.m}, r={R})")
    plt.savefig(f'plots/task_1_1_step_histogram_R{R}.png', dpi=300, bbox_inches='tight')
    if args.show:
        plt.show()
    else:
        plt.close()
