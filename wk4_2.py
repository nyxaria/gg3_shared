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

import models
from utils import calculate_psth
from utils import calculate_fano_factor

T = 1000
N_TRIALS = 5000
PSTH_BIN_WIDTH_MS = 25

# step model plots

fig1 = plt.figure()
fig2 = plt.figure()
fig3 = plt.figure()

for g in np.array([1,3,5]):
    model = models.StepModel(m=T/2, r=2, isi_gamma_shape=g)
    spikes, jumps, _ = model.simulate(Ntrials=N_TRIALS, T=T)
    
    plt.figure(fig1.number)
    plt.hist(jumps, alpha=0.5, label=f'm={model.m}, r={model.r}, Shape={g}')

    # spike raster
    plt.figure(figsize=(10, 6))
    for trial_idx in range(10):
        spike_times_trial = np.where(spikes[trial_idx, :] > 0)[0]
        plt.plot(spike_times_trial, np.ones_like(spike_times_trial) * trial_idx, '|', color='black', markersize=5)
        
        plt.plot(jumps[trial_idx], trial_idx, 'ro', markersize=5, label='Jump Time' if trial_idx == 0 else "")

    plt.xlabel("Time (ms)")
    plt.ylabel("Trial Number")
    plt.title(f"Step Model - Spike Raster (m={model.m}, r={model.r}, Shape={g})")
    plt.ylim(-1, 10)
    if N_TRIALS > 0: plt.legend()
    plt.savefig(f'plots/task_4_2_step_raster_shape={g}.png', dpi=300, bbox_inches='tight')
    
    # psth
    psth_time_bins_ms, psth_values_hz = calculate_psth(spikes, T, PSTH_BIN_WIDTH_MS)
    plt.figure(fig2.number)
    plt.plot(psth_time_bins_ms, psth_values_hz, label=f'm={model.m}, r={model.r}, Shape={g}')
    
    # fano
    spikes, _, _ = model.simulate(Ntrials=50000, T=T)
    fano_time_bins_ms, fano_factors = calculate_fano_factor(spikes, T, PSTH_BIN_WIDTH_MS)
    plt.figure(fig3.number)
    plt.plot(fano_time_bins_ms, fano_factors, label=f'm={model.m}, r={model.r}, Shape={g}')

plt.figure(fig1.number)
plt.xlabel('Time (ms)')
plt.ylabel('Frequency')
plt.title('Jump Times Histogram')
plt.legend()
plt.grid()
filename = f"plots/task_4_2_step_hist.png"
plt.savefig(filename)

plt.figure(fig2.number)
plt.xlabel('Time (ms)')
plt.ylabel('Firing Rate (Hz)')
plt.title('Step Model PSTH For Different Gamma Shapes')
plt.legend()
plt.grid()
filename = f"plots/task_4.2_step_psth.png"
plt.savefig(filename)

plt.figure(fig3.number)
plt.plot(fano_time_bins_ms, np.ones(len(fano_time_bins_ms)), 'k--', label='Poisson (Fano=1)')
plt.xlabel('Time (ms)')
plt.ylabel('Fano Factor')
plt.title('Step Model Fano Factor For Different Gamma Shapes')
plt.legend()
plt.grid()
filename = f"plots/task_4_2_step_fano.png"
plt.savefig(filename)
print('ramp models')
# ramp model plots

fig4 = plt.figure()
fig5 = plt.figure()
fig6 = plt.figure()

for g in np.array([1,3,5]):
    model = models.RampModel(isi_gamma_shape=g)
    spikes, xs, _ = model.simulate(Ntrials=N_TRIALS, T=T)

    # calculate bound hitting times
    bound_hitting_times = []
    for trial_idx in range(N_TRIALS):
        trial_xs = xs[trial_idx, :]
        hit_times = np.where(trial_xs >= 1.0)[0]
        if len(hit_times) > 0:
            bound_hitting_times.append(hit_times[0])
        else:
            bound_hitting_times.append(T)

    # histogram bound hitting times
    plt.figure(fig4.number)
    plt.hist(bound_hitting_times, alpha=0.5, label=f'β={model.beta}, σ={model.sigma}, Shape={g}')


    # spike raster
    plt.figure(figsize=(10, 6))
    for trial_idx in range(10):
        spike_times_trial = np.where(spikes[trial_idx, :] > 0)[0]
        plt.plot(spike_times_trial, np.ones_like(spike_times_trial) * trial_idx, '|', color='black', markersize=5)
        plt.plot(bound_hitting_times[trial_idx], trial_idx, 'ro', markersize=5, 
                     label='Bound Hit Time' if trial_idx == 0 else "")
        
    plt.xlabel("Time (ms)")
    plt.ylabel("Trial Number")
    plt.title(f"Ramp Model - Spike Raster (β={model.beta}, σ={model.sigma}, Shape={g})")
    plt.ylim(-1, 10)
    if N_TRIALS > 0: plt.legend()
    plt.savefig(f'plots/task_4_2_ramp_raster_shape={g}.png', dpi=300, bbox_inches='tight')

    # psth
    psth_time_bins_ms, psth_values_hz = calculate_psth(spikes, T, PSTH_BIN_WIDTH_MS)
    plt.figure(fig5.number)
    plt.plot(psth_time_bins_ms, psth_values_hz, label=f'β={model.beta}, σ={model.sigma}, Shape={g}')
    
    # fano
    spikes, _, _ = model.simulate(Ntrials=50000, T=T)
    fano_time_bins_ms, fano_factors = calculate_fano_factor(spikes, T, PSTH_BIN_WIDTH_MS)
    plt.figure(fig6.number)
    plt.plot(fano_time_bins_ms, fano_factors, label=f'β={model.beta}, σ={model.sigma}, Shape={g}')

plt.figure(fig4.number)
plt.xlabel('Time (ms)')
plt.ylabel('Frequency')
plt.title('Hit Times Histogram')
plt.legend()
plt.grid()
filename = f"plots/task_4_2_ramp_hist.png"
plt.savefig(filename)

plt.figure(fig5.number)
plt.xlabel('Time (ms)')
plt.ylabel('Firing Rate (Hz)')
plt.title('Ramp Model PSTH For Different Gamma Shapes')
plt.legend()
plt.grid()
filename = f"plots/task_4_2_ramp_psth.png"
plt.savefig(filename)

plt.figure(fig6.number)
plt.plot(fano_time_bins_ms, np.ones(len(fano_time_bins_ms)), 'k--', label='Poisson (Fano=1)')
plt.xlabel('Time (ms)')
plt.ylabel('Fano Factor')
plt.title('Ramp Model Fano Factor For Different Gamma Shapes')
plt.legend()
plt.grid()
filename = f"plots/task_4_2_ramp_fano.png"
plt.savefig(filename)