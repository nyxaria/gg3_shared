import numpy as np
import matplotlib.pyplot as plt
import models
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--show', action='store_true', help='Show plots interactively')
args = parser.parse_args()

N_TRIALS = 50  # number of trials to simulate
T_DURATION = 1000  # duration of each trial in time-steps or ms

beta_values = [0.5, 1, 2, 4]  # drift rate
sigma_values = [0.3]  # noise/diffusion

for beta in beta_values:
    for sigma in sigma_values:
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

        # spike raster
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

        # trajectories
        plt.figure(figsize=(10, 6))
        num_trajectories_to_plot = min(N_TRIALS, 10)
        time_axis = np.arange(T_DURATION)

        for trial_idx in range(num_trajectories_to_plot):
            plt.plot(time_axis, xs_ramp[trial_idx, :], label=f'Trial {trial_idx+1}' if trial_idx < 5 else None)

        plt.xlabel("Time (ms)")
        plt.ylabel("Latent Variable x_t")
        plt.title(f"Ramp Model - Latent Variable Trajectories (β={beta}, σ={sigma})")
        plt.axhline(1.0, color='r', linestyle='--', label='Boundary x_t=1')
        if num_trajectories_to_plot > 0: plt.legend(loc='lower right')
        plt.savefig(f'plots/task_1_1_ramp_trajectories_B{beta}_S{sigma}.png', dpi=300, bbox_inches='tight')
        if args.show:
            plt.show()
        else:
            plt.close()

        # bound hitting times histogram
        plt.figure(figsize=(8, 5))
        if bound_hitting_times:
            plt.hist(bound_hitting_times, bins=50, color='lightgreen', edgecolor='black', range=(0, T_DURATION))
        plt.xlabel("Time (ms)")
        plt.ylabel("Frequency (Hz)")
        plt.title(f"Ramp Model - Histogram of Bound-Hitting Times (β={beta}, σ={sigma})")
        plt.savefig(f'plots/task_1_1_ramp_bound_hitting_B{beta}_S{sigma}.png', dpi=300, bbox_inches='tight')
        if args.show:
            plt.show()
        else:
            plt.close()

        print(f"Mean bound-hitting time: {np.mean(bound_hitting_times):.2f} ms")