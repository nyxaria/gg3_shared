import numpy as np
import matplotlib.pyplot as plt
import models
import models_hmm
import os
from tqdm import tqdm
import pickle
import concurrent.futures

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

def _run_single_simulation(args):
    """Helper for running one simulation instance for parallel execution."""
    model_class, sim_func, m, r, n_trials, T, bin_edges = args
    model_instance = model_class(m=m, r=r)
    sim_function = getattr(model_instance, sim_func)
    _, jumps, _ = sim_function(Ntrials=n_trials, T=T)
    counts, _ = np.histogram(jumps, bins=bin_edges, density=True)
    return counts

def compare_jump_time_histograms(m, r, n_trials, n_datasets, T):
    """
    Compares the jump time histograms of the week 1 model and the three
    HMM approximations, using parallel processing and caching.
    """
    
    models_to_compare = {
        "Week 1 Model": {
            "model_class": models.StepModel,
            "sim_func": "simulate",
            "plot_type": "line"
        },
        "HMM (2-state)": {
            "model_class": models_hmm.StepModelHMM,
            "sim_func": "simulate_2state",
        },
        "HMM (r+1-state)": {
            "model_class": models_hmm.StepModelHMM,
            "sim_func": "simulate_exact",
        },
        "HMM (Inhomogeneous 2-state)": {
            "model_class": models_hmm.StepModelHMM,
            "sim_func": "simulate_exact_2state",
        }
    }
    
    cache_dir = os.path.join("plots", "cache")
    os.makedirs(cache_dir, exist_ok=True)
    
    bin_edges = np.linspace(0, T, 51)
    bin_width = bin_edges[1] - bin_edges[0]
    
    all_avg_hists = {}

    for name, details in models_to_compare.items():
        print(f"Processing {name}...")
        
        safe_name = name.replace(" ", "_").replace("(", "").replace(")", "").replace("+", "")
        cache_filename = os.path.join(cache_dir, f"task_2_2_hists_{safe_name}_m{m}_r{r}_T{T}_N{n_trials}_D{n_datasets}.pickle")

        if os.path.exists(cache_filename):
            with open(cache_filename, 'rb') as f:
                all_avg_hists[name] = pickle.load(f)
            print(f"  Loaded from cache: {cache_filename}")
        else:
            tasks = [(details["model_class"], details["sim_func"], m, r, n_trials, T, bin_edges) for _ in range(n_datasets)]
            
            dataset_hist_counts = []
            with concurrent.futures.ProcessPoolExecutor() as executor:
                results = list(tqdm(executor.map(_run_single_simulation, tasks), total=len(tasks), desc=f"  Datasets for {name}"))
                dataset_hist_counts.extend(results)

            avg_hist = np.mean(dataset_hist_counts, axis=0)
            all_avg_hists[name] = avg_hist
            
            with open(cache_filename, 'wb') as f:
                pickle.dump(avg_hist, f)
            print(f"  Saved to cache: {cache_filename}")

    plt.figure(figsize=(12 * 0.7, 8 * 0.7))
    
    for name, avg_hist in all_avg_hists.items():
        if models_to_compare[name].get('plot_type') == "line":
            bin_centers = bin_edges[:-1] + bin_width / 2

            plt.plot(bin_centers, avg_hist, alpha=1, label=name, color='black')
        else:
            plt.bar(bin_edges[:-1], avg_hist, width=bin_width, alpha=0.5, label=name, align='edge')

    plt.xlabel('Jump Time (ms)')
    plt.ylabel('Probability Density')
    plt.title(f'Comparison of Jump Time Distributions\n(m={m}, r={r}, N_trials={n_trials}, N_datasets={n_datasets})')
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    savename = f'plots/task_2_2_histogram_comparison_m{m}_r{r}_line.png'
    os.makedirs('plots', exist_ok=True)
    plt.savefig(savename, dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    T_DURATION = 1000
    N_TRIALS = 1000
    N_DATASETS = 20
    
    M = 500
    R = 100
    
    compare_jump_time_histograms(
        m=M,
        r=R,
        n_trials=N_TRIALS,
        n_datasets=N_DATASETS,
        T=T_DURATION
    )

import models
import os
import pickle
from tqdm import tqdm
import concurrent.futures

def _run_single_step_grid_point(args):
    """Helper function to run simulation for a single parameter grid point."""
    m, r, n_trials, T, n_datasets = args
    
    rmses = []
    signed_errors = []

    for _ in range(n_datasets):
        week1_model = models.StepModel(m=m, r=r)
        hmm_model = models_hmm.StepModelHMM(m=m, r=r)

        _, _, rates_week1 = week1_model.simulate(Ntrials=n_trials, T=T, get_rate=True)
        _, _, rates_hmm = hmm_model.simulate_exact_2state(Ntrials=n_trials, T=T)

        avg_rates_week1 = np.mean(rates_week1, axis=0)
        avg_rates_hmm = np.mean(rates_hmm, axis=0)

        rmse = np.sqrt(np.mean((avg_rates_hmm - avg_rates_week1)**2))
        signed_error = np.mean(avg_rates_hmm - avg_rates_week1)
        
        rmses.append(rmse)
        signed_errors.append(signed_error)

    return np.mean(rmses), np.mean(signed_errors)

def analyze_step_model_accuracy_heatmap(m_vals, r_vals, n_trials, T, n_datasets):
    """
    Analyzes the accuracy of the Inhomogeneous 2-state HMM against the Week 1 model
    across a parameter grid and plots the results as heatmaps.
    """
    print("\n--- Running Step Model HMM Accuracy Analysis ---")
    cache_dir = os.path.join("plots", "cache")
    os.makedirs(cache_dir, exist_ok=True)
    
    m_len = len(m_vals)
    r_len = len(r_vals)
    cache_filename = os.path.join(cache_dir, f"task_2_2_heatmap_errors_m{m_len}_r{r_len}_T{T}_N{n_trials}_D{n_datasets}.pickle")

    if os.path.exists(cache_filename):
        with open(cache_filename, 'rb') as f:
            errors = pickle.load(f)
        print(f"Loaded errors from cache: {cache_filename}")
    else:
        tasks = []
        for r in r_vals:
            for m in m_vals:
                tasks.append((m, r, n_trials, T, n_datasets))
        
        results = []
        with concurrent.futures.ProcessPoolExecutor() as executor:
            results = list(tqdm(executor.map(_run_single_step_grid_point, tasks), total=len(tasks), desc="Processing grid points"))
        
        rmse_errors = np.array([res[0] for res in results]).reshape(len(r_vals), len(m_vals))
        signed_errors = np.array([res[1] for res in results]).reshape(len(r_vals), len(m_vals))
        errors = {'rmse': rmse_errors, 'signed': signed_errors}
        
        with open(cache_filename, 'wb') as f:
            pickle.dump(errors, f)
        print(f"Saved errors to cache: {cache_filename}")

    # --- Plotting ---
    delta_m = m_vals[1] - m_vals[0] if len(m_vals) > 1 else 0
    extent = [m_vals[0] - delta_m / 2, m_vals[-1] + delta_m / 2, r_vals[0] - 0.5, r_vals[-1] + 0.5]

    # Plot RMSE
    plt.figure(figsize=(10, 8))
    im = plt.imshow(errors['rmse'], aspect='auto', origin='lower', extent=extent, cmap='viridis')
    plt.colorbar(im, label="RMSE of mean firing rate")
    plt.xlabel("m (mean jump time)")
    plt.ylabel("r (jump time variability)")
    plt.yticks(r_vals)
    plt.title(f"Step Model HMM vs. Week 1 Model Accuracy (RMSE)\n(N_trials={n_trials}, N_datasets={n_datasets})")
    plt.savefig('plots/task_2_2_heatmap_rmse.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Plot Signed Error
    plt.figure(figsize=(10, 8))
    signed_errors_matrix = errors['signed']
    max_abs_error = np.max(np.abs(signed_errors_matrix))
    im = plt.imshow(signed_errors_matrix, aspect='auto', origin='lower', extent=extent, cmap='coolwarm', vmin=-max_abs_error, vmax=max_abs_error)
    plt.colorbar(im, label="Signed Mean Error (HMM - Week 1)")
    plt.xlabel("m (mean jump time)")
    plt.ylabel("r (jump time variability)")
    plt.yticks(r_vals)
    plt.title(f"Step Model HMM vs. Week 1 Model Accuracy (Signed Error)\n(N_trials={n_trials}, N_datasets={n_datasets})")
    plt.savefig('plots/task_2_2_heatmap_signed_error.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    T_DURATION = 1000
    N_TRIALS_HEATMAP = 1000
    N_GRID_POINTS_M = 15
    N_DATASETS_HEATMAP = 5 # Average over 5 datasets for stability

    m_values = np.linspace(1, T_DURATION * 3 / 4, N_GRID_POINTS_M)
    r_values = np.arange(1, 7)
    
    analyze_step_model_accuracy_heatmap(
        m_vals=m_values,
        r_vals=r_values,
        n_trials=N_TRIALS_HEATMAP,
        T=T_DURATION,
        n_datasets=N_DATASETS_HEATMAP
    )