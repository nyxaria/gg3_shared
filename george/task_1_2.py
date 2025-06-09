import numpy as np
import matplotlib.pyplot as plt
import models
import argparse
from utils import calculate_psth

parser = argparse.ArgumentParser()
parser.add_argument('--show', action='store_true')
args = parser.parse_args()

T_DURATION_MS = 1000
PSTH_BIN_WIDTH_MS = 25

def calculate_psth_fluctuations(spike_trains, t_duration_ms, bin_width_ms):
    n_trials = spike_trains.shape[0]

    num_bins = int(t_duration_ms / bin_width_ms)

    all_trials_binned_counts = np.zeros((n_trials, num_bins))

    for trial_i in range(n_trials):
        for bin_idx in range(num_bins):
            start_time = bin_idx * bin_width_ms
            end_time = (bin_idx + 1) * bin_width_ms
            all_trials_binned_counts[trial_i, bin_idx] = np.sum(spike_trains[trial_i, start_time:end_time])
            
    std_counts_per_bin = np.std(all_trials_binned_counts)
    se_counts_per_bin = std_counts_per_bin / np.sqrt(n_trials)
    se_rate_per_bin_hz = se_counts_per_bin / (bin_width_ms / 1000.0)
    
    average_se_of_psth = np.mean(se_rate_per_bin_hz)
    return average_se_of_psth

def analyze_psth_fluctuations(show_plot):
    n_trials_list = np.array([20, 50, 100, 200, 500, 1000, 2000])
    avg_se_list = []

    model = models.RampModel(beta=1, sigma=0.3)

    for n_trials in n_trials_list:
        spikes, _, _ = model.simulate(Ntrials=n_trials, T=T_DURATION_MS)
        avg_se = calculate_psth_fluctuations(spikes, T_DURATION_MS, PSTH_BIN_WIDTH_MS)
        avg_se_list.append(avg_se)

    avg_se_list = np.array(avg_se_list)

    plt.figure(figsize=(10, 6))
    plt.plot(n_trials_list, avg_se_list, 'o-', label='Average SE of PSTH')
    
    # filter for NaN
    valid_indices = ~np.isnan(avg_se_list)
    if np.any(valid_indices) and sum(valid_indices) > 1:
        # use the first point to scale a 1/sqrt(N) curve
        scale_factor = avg_se_list[valid_indices][0] * np.sqrt(n_trials_list[valid_indices][0])
        plt.plot(n_trials_list, scale_factor / np.sqrt(n_trials_list), '--', label='1/√N (scaled)')

    plt.xlabel("Number of Trials (N)")
    plt.ylabel("Average Standard Error of PSTH (Hz)")
    plt.title("PSTH Fluctuation vs. Number of Trials")
    plt.grid(True, alpha=0.5)
    plt.legend()
    plt.savefig('plots/task_1_2_psth_fluctuations.png', dpi=300, bbox_inches='tight')
    if show_plot:
        plt.show()
    else:
        plt.close()


def evaluate_psth_with_n_trials(show_plot, model_type):
    n_trials_to_plot = [100, 2000]
    N_TRIALS_REFERENCE = 50000

    if model_type.lower() == 'ramp':
        model = models.RampModel(beta=1.5, sigma=0.4)
        model_desc = f"Ramp Model (β={1.5}, σ={0.4})"
    else:
        model = models.StepModel(m=T_DURATION_MS/2, r=20)
        model_desc = f"Step Model (m={T_DURATION_MS/2}, r={20})"

    plt.figure(figsize=(12, 7))
    
    spikes_ref, _, _ = model.simulate(Ntrials=N_TRIALS_REFERENCE, T=T_DURATION_MS)
    time_bins, psth_ref_raw = calculate_psth(spikes_ref, T_DURATION_MS, PSTH_BIN_WIDTH_MS)
    plt.plot(time_bins, psth_ref_raw, label=f'Reference PSTH (N={N_TRIALS_REFERENCE})', color='black', linestyle='--', linewidth=2)

    for i, n_trials in enumerate(n_trials_to_plot):
        spikes, _, _ = model.simulate(Ntrials=n_trials, T=T_DURATION_MS)
        current_time_bins, psth_raw = calculate_psth(spikes, T_DURATION_MS, PSTH_BIN_WIDTH_MS)
        plt.plot(current_time_bins, psth_raw, label=f'PSTH (N={n_trials})', alpha=0.8)

    plt.xlabel("Time (ms)")
    plt.ylabel("Firing Rate (Hz)")
    plt.title(f"PSTH with Varying Number of Trials\n{model_desc}")
    plt.legend()
    plt.grid(True, alpha=0.5)
    
    plot_filename = f"plots/task_1_2_psth_n_trials_{model_type.lower()}.png"
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_psths(params):
    plt.figure(figsize=(12, 7))
    

    for param in params:
        model_type = "Unknown"
        model = None
        
        if 'beta' in param and 'sigma' in param:
            model_type = "Ramp"
            model = models.RampModel(beta=param['beta'], sigma=param['sigma'])
        else:
            model_type = "Step"
            model = models.StepModel(m=param['m'], r=param['r'])

        spikes, _, _ = model.simulate(Ntrials=1000, T=100)
        
        time_bins, psth_hz = calculate_psth(spikes, 100, PSTH_BIN_WIDTH_MS//5)
        
        label_params = ", ".join([f"{key}={value}" for key, value in param.items()])
        plt.plot(time_bins, psth_hz, marker='.', linestyle='-', markersize=5, label=f'{model_type}: {label_params}')

    plt.title(f'PSTH Comparison for indistuingishable case')
    plt.xlabel(f'Time (ms)')
    plt.ylabel('PSTH (Hz)')
    plt.legend()
    plt.grid(True, alpha=0.5)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    analyze_psth_fluctuations(args.show)
    
    evaluate_psth_with_n_trials(args.show, 'ramp')
    evaluate_psth_with_n_trials(args.show, 'step')

    param_sets = [
        {"beta": 1.5, "sigma": 0.27},
        {"m": 30, "r": 2.45},
    ]

    plot_psths(param_sets)
