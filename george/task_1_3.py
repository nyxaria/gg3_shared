import numpy as np
import matplotlib.pyplot as plt
import models
import argparse
from utils import calculate_fano_factor, generate_model_parameters, generate_random_model_parameters

RUNNING_IN_JUPYTER = True
if not RUNNING_IN_JUPYTER:
    parser = argparse.ArgumentParser()
    parser.add_argument('--show', action='store_true')
    args = parser.parse_args()
else:
    # dont have command line args in jupyter, mock it
    class ArgsMock:
        show = True
    args = ArgsMock()

T_DURATION_MS = 1000
FANO_BIN_WIDTH_MS = 50
N_TRIALS_FANO = 5000

def analyze_fano_factors(params, label, show_plot):

    model_type = 'ramp' if all('beta' in p for p in params) else ('step' if all('m' in p for p in params) else 'mixed')

    plt.figure(figsize=(12, 7))
    for params in params:
        if 'beta' in params:
            model = models.RampModel(beta=params['beta'], sigma=params['sigma'])
        elif 'm' in params:
            model = models.StepModel(m=params['m'], r=params['r'])
        spikes, _, _ = model.simulate(Ntrials=N_TRIALS_FANO, T=T_DURATION_MS)
        time_bins, fano_vals = calculate_fano_factor(spikes, T_DURATION_MS, FANO_BIN_WIDTH_MS)
        # Fano stats
        valid_fano = fano_vals[~np.isnan(fano_vals)]
        mean_fano = np.mean(valid_fano)
        var_fano = np.var(valid_fano)
        start_fano = valid_fano[0] if len(valid_fano) > 0 else np.nan
        description = f"β={params['beta']}, σ={params['sigma']}" if 'beta' in params else f"m={params['m']}, r={params['r']}"
        if show_plot:
            print(f"Fano factor stats for {description}: mean: {mean_fano:.3f} starting val: {start_fano:.3f} variance: {var_fano:.3f}")
        
        plt.plot(time_bins, fano_vals, label=f"{'ramp' if 'beta' in params else 'step'}: {description}", alpha=0.8)
   
    plt.axhline(1, color='k', linestyle='--', label='Poisson (Fano=1)')
    plt.xlabel(f"Time (ms)")
    plt.ylabel("Fano Factor")
    model_label = "for Ramp Model" if model_type == "ramp" else "for Step Model" if model_type == "step" else ""
    plt.title(f"Fano Factor vs. Time {model_label} (N_trials={N_TRIALS_FANO})")
    plt.legend()
    plt.grid(True, alpha=0.5)
    plt.ylim(0, 2.5)
    plt.savefig('plots/task_1_3_fano_factor_{label}.png', dpi=300, bbox_inches='tight')
    if show_plot:
        plt.show()
    else:
        plt.close()


def generate_random_model_parameters():
    model_type = np.random.choice(['ramp', 'step'])
    if model_type == 'ramp':
        beta = np.random.uniform(0.1, 50.0)
        sigma = np.random.uniform(0.1, 5.0)
        params = {'beta': beta, 'sigma': sigma}
    else:
        m = np.random.uniform(T_DURATION_MS * 0.1, T_DURATION_MS * 0.9)
        r = np.random.uniform(0.1, 50.0)
        params = {'m': m, 'r': r}

    return params

def plot_step_model_trials(m_param, r_param, trial_counts, T_duration, show_plot):
    """
    Plots the activity of a StepModel for different numbers of trials.
    - For N=1, plots the spike train of a single trial.
    - For N in (1, 300], plots a raster of spike trains.
    - For N > 300 (large N), plots the Peri-Stimulus Time Histogram (PSTH).
    """
    model = models.StepModel(m=m_param, r=r_param)
    
    num_plots = len(trial_counts)
    # Ensure axes is a flat array for easy indexing, even if num_plots is 1
    fig, axes = plt.subplots(num_plots, 1, figsize=(12, 3.5 * num_plots), sharex=True, squeeze=False)
    axes = axes.flatten() 

    fig.suptitle(f"Step Model Activity (m={m_param}ms, effect of r={r_param})", fontsize=16)

    for i, N in enumerate(trial_counts):
        # Assuming model.simulate returns:
        # spikes: list of arrays (spike times per trial for each of N trials)
        # rates_psth: array (average firing rate over time, i.e., PSTH)
        # binned_spikes: 2D array (trial x time_bin) of spike counts (currently unused here)
        spikes, rates_psth, _ = model.simulate(Ntrials=N, T=T_duration)
        
        ax = axes[i]
        
        # Plot vertical line for step time 'm'
        ax.axvline(m_param, color='tomato', linestyle='--', linewidth=1.5, label=f'Step at t={m_param}ms', zorder=1)

        if N == 1:
            current_title = f"Single Trial (N=1)"
            # Based on ValueError, assume 'spikes' is a direct np.ndarray for N=1
            if isinstance(spikes, np.ndarray) and spikes.size > 0:
                ax.eventplot(spikes, lineoffsets=1, linelengths=0.8, colors='k', zorder=2) # Plot 'spikes' directly
                ax.set_yticks([1])
                ax.set_yticklabels(['Trial 1'])
            else: # No spikes, or not an ndarray (e.g. None), or an empty ndarray
                ax.text(0.5, 0.5, 'No spikes', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
                ax.set_yticks([])
            ax.set_ylim(0.5, 1.5)
        elif N > 1 and N <= 300: # Raster plot for N (e.g., 300)
            current_title = f"Raster Plot (N={N})"
            # Check if spikes is a NumPy array and contains any finite values (actual spikes)
            if spikes is not None and np.any(np.isfinite(spikes)):
                 # eventplot can take a 2D array, treating rows as trials.
                 # Ensure lineoffsets match the number of trials, which should be N.
                 ax.eventplot(spikes, colors='k', linelengths=0.8, lineoffsets=np.arange(1, N + 1), zorder=2)
                 ax.set_ylim(0.5, N + 0.5)
                 ax.set_ylabel("Trial Number")
            else:
                 ax.text(0.5, 0.5, 'No spikes', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
                 ax.set_yticks([]) 
                 ax.set_ylabel("") 
        elif N > 300: # PSTH for large N (e.g., 10000)
            current_title = f"PSTH (N={N})"
            time_points_psth = np.linspace(0, T_duration, len(rates_psth), endpoint=True)
            ax.plot(time_points_psth, rates_psth, color='dodgerblue', linewidth=1.5, label='Avg. Firing Rate', zorder=2)
            ax.set_ylabel("Firing Rate (spikes/s)")
            ax.grid(True, linestyle=':', alpha=0.7)
            ax.legend(loc='best') # Legend for PSTH (includes axvline label)

        ax.set_title(current_title)
        ax.set_xlim(0, T_duration)

        # Add legend for the axvline if not the PSTH plot (which already has a combined legend)
        if not (N > 300):
            # Only show legend if there are actual elements labeled (the axvline here)
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                 ax.legend(loc='best')

        if i == num_plots - 1: # Only set xlabel for the bottom subplot
            ax.set_xlabel("Time (ms)")
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust rect for suptitle and ensure layout is tight
    
    plot_filename = f'plots/task_1_3_step_model_trials_m{m_param}_r{r_param}.png'
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"Saved plot to {plot_filename}")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)

def plot_fano_factor_for_step_model_trials(m_param, r_param, trial_counts, T_duration, fano_bin_width_ms, show_plot):
    """
    Calculates and plots Fano factors for a StepModel with varying numbers of trials.
    Each trial count will be represented as a different line on the same plot.
    """
    plt.figure(figsize=(12, 7))
    model_instance = models.StepModel(m=m_param, r=r_param) # Create model once

    for N in trial_counts:
        spikes, _, _ = model_instance.simulate(Ntrials=N, T=T_duration)
        time_bins, fano_vals = calculate_fano_factor(spikes, T_duration, fano_bin_width_ms)
        
        plt.plot(time_bins, fano_vals, label=f"N={N}", alpha=0.8)

        # Optional: Print some stats for each N if desired
        valid_fano = fano_vals[~np.isnan(fano_vals)]
        mean_fano = np.mean(valid_fano) if len(valid_fano) > 0 else np.nan
        print(f"For N={N}, m={m_param}, r={r_param}: Mean Fano Factor: {mean_fano:.3f} (using {len(valid_fano)} valid points)")

    plt.axhline(1, color='k', linestyle='--', label='Poisson (Fano=1)')
    # plt.axvline(m_param, color='tomato', linestyle=':', linewidth=1.5, label=f'Step at t={m_param}ms')
    plt.xlabel(f"Time (ms)")
    plt.ylabel("Fano Factor")
    plt.title(f"Fano Factor vs. Time for Step Model (m={m_param}, r={r_param}) across Trial Counts")
    plt.legend()
    plt.grid(True, alpha=0.5)
    plt.ylim(0, plt.ylim()[1] * 1.1 if plt.ylim()[1] > 2 else 2.5) # Adjust y-lim, ensure min 2.5

    plot_filename = f'plots/task_1_3_fano_step_trials_m{m_param}_r{r_param}.png'
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"Saved Fano factor plot to {plot_filename}")

    if show_plot:
        plt.show()
    else:
        plt.close()

if __name__ == "__main__":
    
    # Call the new function to plot step model trials
    step_m_param = 500
    step_r_param = 5
    trial_counts_for_step_plot = [1, 300, 10000]
    # plot_step_model_trials(m_param=step_m_param, 
    #                        r_param=step_r_param, 
    #                        trial_counts=trial_counts_for_step_plot, 
    #                        T_duration=T_DURATION_MS, 
    #                        show_plot=args.show)

    # Call the new function to plot Fano factors for step model with different trial counts
    fano_trial_counts = [300, 10000] # Using a range of N values
    plot_fano_factor_for_step_model_trials(m_param=step_m_param, # Use same m, r as above
                                           r_param=step_r_param,
                                           trial_counts=fano_trial_counts,
                                           T_duration=T_DURATION_MS,
                                           fano_bin_width_ms=FANO_BIN_WIDTH_MS,
                                           show_plot=args.show)

    params = {
        # "random_1": [generate_random_model_parameters(), generate_random_model_parameters()],
        # "random_2": [generate_random_model_parameters(), generate_random_model_parameters()],
        # "random_3": [generate_random_model_parameters(), generate_random_model_parameters()],
        # "random_4": [generate_random_model_parameters(), generate_random_model_parameters()],
        # "random_5": [generate_random_model_parameters(), generate_random_model_parameters()],
        # "random_6": [generate_random_model_parameters(), generate_random_model_parameters()],
        # "random_7": [generate_random_model_parameters(), generate_random_model_parameters()],
        # "random_8": [generate_random_model_parameters(), generate_random_model_parameters()],
        "ramp_sigma": [
            {'beta': 0.5, 'sigma': 0.1},
            {'beta': 0.5, 'sigma': 0.4},
            {'beta': 0.5, 'sigma': 0.8}
        ],
        "ramp_beta": [
            {'beta': 0.5, 'sigma': 0.2},
            {'beta': 10, 'sigma': 0.2},
            {'beta': 50, 'sigma': 0.2}
        ],
        "step_r": [
            {'m': T_DURATION_MS / 2, 'r': 0.1},
            {'m': T_DURATION_MS / 2, 'r': 1},
            {'m': T_DURATION_MS / 2, 'r': 20},
            {'m': T_DURATION_MS / 2, 'r': 100}
        ],
        "step_m": [
            {'m': T_DURATION_MS / 4, 'r': 5},
            {'m': T_DURATION_MS / 2, 'r': 5},
            {'m': T_DURATION_MS * .75, 'r': 5}
        ],
    # }
    # params = {
        "indistinguishable_1": [
            {'beta': 1.0, 'sigma': 0.3},
            {'m': 50, 'r': 2.0}
        ],
        "indistinguishable_2": [
            {'beta': 1.5, 'sigma': 0.27},
            {'m': 30, 'r': 2.45}
        ],
        "indistinguishable_3": [
            {'beta': 1.8, 'sigma': 0.43}, 
            {'m': 26, 'r': 2.15}
        ]
    }
    print(params)
    for param_type, param_list in params.items():
        print(param_type)
        analyze_fano_factors(param_list, param_type, args.show) 
