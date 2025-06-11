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

N_TRIALS = 500  # number of trials to simulate
T_DURATION = 1000  # duration of each trial in time-steps or ms

M_values = [T_DURATION / 3, T_DURATION / 3, T_DURATION * 2 / 3] # mean jump time
R_values = [10, 100, 100] # jump time variability 

all_jumps = {}
all_avg_rates = {}

for M, R in zip(M_values, R_values):
    N_TRIALS = 50
    M = 500
    R = 100
    print(f"\nTesting M={M}, R={R}")
    step_model = models.StepModel(m=M, r=R)

    spikes_step, jumps_step, rates_step = step_model.simulate(Ntrials=N_TRIALS, T=T_DURATION, get_rate=True)

    all_jumps[(M, R)] = jumps_step
    avg_rates = np.mean(rates_step, axis=0)
    all_avg_rates[(M, R)] = avg_rates

    # spike raster
    plt.figure(figsize=(10, 6))
    for trial_idx in range(N_TRIALS):
        spike_times_trial = np.where(spikes_step[trial_idx, :] > 0)[0]
        plt.plot(spike_times_trial, np.ones_like(spike_times_trial) * trial_idx, '|', color='black', markersize=5)
        
        plt.plot(jumps_step[trial_idx], trial_idx, 'ro', markersize=5, label='Jump Time' if trial_idx == 0 else "")

    plt.xlabel("Time (ms)")
    plt.ylabel("Trial Number")
    plt.title(f"Step Model - Spike Raster (m={M}, r={R})")
    plt.ylim(-1, N_TRIALS)
    if N_TRIALS > 0: plt.legend()
    plt.savefig(f'plots/task_1_1_step_raster_M{M}_R{R}.png', dpi=300, bbox_inches='tight')
    if args.show:
        plt.show()
    else:
        plt.close()
    break
N_TRIALS = 500  # number of trials to simulate

# Combined histogram of jump times
plt.figure(figsize=(8, 5))
for (M, R), jumps in all_jumps.items():
    plt.hist(jumps, bins=50, range=(0, T_DURATION), density=True, alpha=0.5, label=f'm={int(M)}, r={R}')
plt.xlabel("Time (ms)")
plt.ylabel("Frequency (Hz)")
plt.title("Step Model - Histogram of Jump Times")
plt.legend()
plt.savefig('plots/task_1_1_step_histogram_combined.png', dpi=300, bbox_inches='tight')
if args.show:
    plt.show()
else:
    plt.close()

# Combined average firing rate plot
plt.figure(figsize=(10, 6))
time_axis = np.arange(T_DURATION)
for (M, R), avg_rates in all_avg_rates.items():
    plt.plot(time_axis, avg_rates, label=f'm={int(M)}, r={R}')

plt.xlabel("Time (ms)")
plt.ylabel("Average Firing Rate (Hz)")
plt.title("Step Model - Average Firing Rate")
plt.legend(loc='upper left')
plt.savefig('plots/task_1_1_step_avg_rates_combined.png', dpi=300, bbox_inches='tight')
if args.show:
    plt.show()
else:
    plt.close()
