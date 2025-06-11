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
parser.add_argument('--show', action='store_true', help='Show plots interactively')
args = parser.parse_args()

N_TRIALS = 100  # number of trials to simulate
T_DURATION = 1000  # duration of each trial in time-steps or ms

beta_values = [4, 4, 1]  # drift rate
sigma_values = [0.1, 0.4, 0.1]  # noise/diffusion

all_bound_hitting_times = {}
all_avg_trajectories = {}

for beta, sigma in zip(beta_values, sigma_values):
    print(f"\nTesting β={beta}, σ={sigma}")
    ramp_model = models.RampModel(beta=beta, sigma=sigma)
    spikes_ramp, xs_ramp, rates_ramp = ramp_model.simulate(Ntrials=N_TRIALS, T=T_DURATION, get_rate=True)

    # calculate bound hitting times
    bound_hitting_times = []
    for trial_idx in range(N_TRIALS):
        trial_xs = xs_ramp[trial_idx, :]
        hit_times = np.where(trial_xs >= 1.0)[0]
        if len(hit_times) > 0:
            bound_hitting_times.append(hit_times[0])
        else:
            bound_hitting_times.append(T_DURATION)
    
    all_bound_hitting_times[(beta, sigma)] = bound_hitting_times
    avg_trajectory = np.mean(xs_ramp, axis=0)
    all_avg_trajectories[(beta, sigma)] = avg_trajectory

    if beta == 4 and sigma == 0.1:
        beta = 2
        sigma = 0.3
        ramp_model = models.RampModel(beta=2, sigma=0.3)
        spikes_ramp, xs_ramp, rates_ramp = ramp_model.simulate(Ntrials=N_TRIALS, T=T_DURATION, get_rate=True)

        # calculate bound hitting times
        bound_hitting_times = []
        for trial_idx in range(N_TRIALS):
            trial_xs = xs_ramp[trial_idx, :]
            hit_times = np.where(trial_xs >= 1.0)[0]
            if len(hit_times) > 0:
                bound_hitting_times.append(hit_times[0])
            else:
                bound_hitting_times.append(T_DURATION)
        # # spike raster
        plt.figure(figsize=(10, 6))
        for trial_idx in range(N_TRIALS):
            spike_times_trial = np.where(spikes_ramp[trial_idx, :] > 0)[0]
            plt.plot(spike_times_trial, np.ones_like(spike_times_trial) * trial_idx, '|', color='black', markersize=5)
            plt.plot(bound_hitting_times[trial_idx], trial_idx, 'ro', markersize=5, 
                    label='Bound Hit Time' if trial_idx == 0 else "")

        plt.xlabel("Time (ms)")
        plt.ylabel("Trial Number")
        plt.title(f"Ramp Model - Spike Raster (β={beta}, σ={sigma})")
        plt.ylim(-1, N_TRIALS)
        if N_TRIALS > 0: plt.legend()
        plt.savefig(f'plots/task_1_1_ramp_raster_B{beta}_S{sigma}.png', dpi=300, bbox_inches='tight')
        if args.show:
            plt.show()
        else:
            plt.close()

    print(f"Mean bound-hitting time: {np.mean(bound_hitting_times):.2f} ms")

# Combined histogram of bound-hitting times
plt.figure(figsize=(10, 6))
for (beta, sigma), bound_hitting_times in all_bound_hitting_times.items():
    if bound_hitting_times:
        plt.hist(bound_hitting_times, bins=50, alpha=0.5, label=f'β={beta}, σ={sigma}', range=(0, T_DURATION))

plt.xlabel("Time (ms)")
plt.ylabel("Frequency (Hz)")
plt.title("Ramp Model - Histogram of Bound-Hitting Times")
plt.legend()
plt.savefig('plots/task_1_1_ramp_bound_hitting_combined.png', dpi=300, bbox_inches='tight')
if args.show:
    plt.show()
else:
    plt.close()

# Combined average trajectories plot
plt.figure(figsize=(10, 6))
time_axis = np.arange(T_DURATION)
for (beta, sigma), avg_trajectory in all_avg_trajectories.items():
    plt.plot(time_axis, avg_trajectory, label=f'β={beta}, σ={sigma}')

plt.xlabel("Time (ms)")
plt.ylabel("Average Latent Variable $x_t$")
plt.title("Ramp Model - Average Latent Variable Trajectories")
plt.axhline(1.0, color='r', linestyle='--', label='Boundary $x_t=1$')
plt.legend(loc='lower right')
plt.savefig('plots/task_1_1_ramp_avg_trajectories_combined.png', dpi=300, bbox_inches='tight')
if args.show:
    plt.show()
else:
    plt.close()