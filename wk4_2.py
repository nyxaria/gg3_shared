import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.lines import Line2D

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

os.makedirs('cache', exist_ok=True)

step_fano_data = []

# step model plots

fig1 = plt.figure()
fig2 = plt.figure()
fig3 = plt.figure()

for g in np.array([1,3,5]):
    model = models.StepModel(m=T/2, r=2, isi_gamma_shape=g)

    cache_fn_main = f'cache/step_m{model.m}_r{model.r}_g{g}_nt{N_TRIALS}_T{T}.npz'
    if os.path.exists(cache_fn_main):
        print(f"Loading cached simulation from {cache_fn_main}")
        data = np.load(cache_fn_main)
        spikes, jumps = data['spikes'], data['jumps']
    else:
        print(f"Running simulation and caching to {cache_fn_main}")
        spikes, jumps, _ = model.simulate(Ntrials=N_TRIALS, T=T)
        np.savez(cache_fn_main, spikes=spikes, jumps=jumps)
    
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
    cache_fn_fano = f'cache/step_m{model.m}_r{model.r}_g{g}_nt50000_T{T}_fano.npz'
    if os.path.exists(cache_fn_fano):
        print(f"Loading cached Fano data from {cache_fn_fano}")
        data = np.load(cache_fn_fano)
        spikes_fano = data['spikes']
    else:
        print(f"Running Fano simulation and caching to {cache_fn_fano}")
        spikes_fano, _, _ = model.simulate(Ntrials=50000, T=T)
        np.savez(cache_fn_fano, spikes=spikes_fano)
        
    fano_time_bins_ms, fano_factors = calculate_fano_factor(spikes_fano, T, PSTH_BIN_WIDTH_MS)
    step_fano_data.append((fano_time_bins_ms, fano_factors, g))
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

ramp_fano_data = []

fig4 = plt.figure()
fig5 = plt.figure()
fig6 = plt.figure()

for g in np.array([1,3,5]):
    model = models.RampModel(beta=2, isi_gamma_shape=g)

    cache_fn_main = f'cache/ramp_b{model.beta}_s{model.sigma}_g{g}_nt{N_TRIALS}_T{T}.npz'
    if os.path.exists(cache_fn_main):
        print(f"Loading cached simulation from {cache_fn_main}")
        data = np.load(cache_fn_main)
        spikes, xs = data['spikes'], data['xs']
    else:
        print(f"Running simulation and caching to {cache_fn_main}")
        spikes, xs, _ = model.simulate(Ntrials=N_TRIALS, T=T)
        np.savez(cache_fn_main, spikes=spikes, xs=xs)

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
    spikes_fano, _, _ = model.simulate(Ntrials=50000, T=T)
        
    fano_time_bins_ms, fano_factors = calculate_fano_factor(spikes_fano, T, PSTH_BIN_WIDTH_MS)
    ramp_fano_data.append((fano_time_bins_ms, fano_factors, g))
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

# --- Combined Fano Factor Plot (Styled like w4_12_plots.py) ---
plt.figure(figsize=(12, 8))
colors = {1: 'blue', 3: 'green', 5: 'red'}

# Plot Step Fano Factors (solid lines)
for bins, factors, g in step_fano_data:
    plt.plot(bins, factors, color=colors[g], linestyle='-')

# Plot Ramp Fano Factors (dashed lines)
for bins, factors, g in ramp_fano_data:
    plt.plot(bins, factors, color=colors[g], linestyle='--')

plt.xlabel('Time (ms)')
plt.ylabel('Fano Factor')
plt.title('Ramp and Step Model Fano Factor vs. Gamma Shapes')
plt.grid(True)
plt.ylim(0, 1.5)

# Create the two-part legend
legend_elements_conditions = [
    Line2D([0], [0], color=colors[1], lw=2, label='Shape=1'),
    Line2D([0], [0], color=colors[3], lw=2, label='Shape=3'),
    Line2D([0], [0], color=colors[5], lw=2, label='Shape=5'),
    Line2D([0], [0], color='k', linestyle=':', lw=2, label='Poisson (Fano=1)')
]

legend_elements_models = [
    Line2D([0], [0], color='gray', linestyle='-', label='Step Model'),
    Line2D([0], [0], color='gray', linestyle='--', label='Ramp Model')
]

# Add the Poisson reference line (plotted last to appear in legend correctly if needed)
if ramp_fano_data:
    last_bins = ramp_fano_data[-1][0]
    plt.plot(last_bins, np.ones(len(last_bins)), 'k:', lw=2)


ax = plt.gca()
leg1 = ax.legend(handles=legend_elements_conditions, title='Gamma Shape', loc='upper right')
ax.add_artist(leg1)
leg2 = ax.legend(handles=legend_elements_models, title='Model Type', loc='center right')


plt.savefig("plots/task_4_2_fano_combined.png")
plt.show()