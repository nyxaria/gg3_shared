import numpy as np
import matplotlib.pyplot as plt
import models
import argparse
import scipy.special
from scipy.optimize import curve_fit
import traceback
from utils import calculate_psth, calculate_fano_factor, generate_model_parameters, classify_model_by_psth
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

if args.show:
    N_DATASETS_PER_CASE = 10
else:
    N_DATASETS_PER_CASE = 1000

T_DURATION_MS = 1000
FANO_BIN_WIDTH_MS = 50
PSTH_BIN_WIDTH_MS = 25
N_TRIALS_CLASSIFY = 400

def classify_model_by_fano(spike_trains, t_duration_ms, fano_bin_width_ms):
    """our strategy here is to detect a peak in the fano factor and then check if it is a step or ramp"""
    _, f_values = calculate_fano_factor(spike_trains, t_duration_ms, fano_bin_width_ms)

    n_smooth = 3
    smoothing_window = np.ones(n_smooth)/n_smooth
    smoothed_f_values = np.convolve(f_values, smoothing_window, mode='valid')

    max_f = np.max(smoothed_f_values)
    idx_max = np.argmax(smoothed_f_values)
    num_fano_bins = len(smoothed_f_values)
    mean_f = np.mean(smoothed_f_values)
    is_peak_central = (idx_max > num_fano_bins * 0.20) and (idx_max < num_fano_bins * 0.80)
    is_peak_prominent = max_f > mean_f * 1.1
    is_peak_at_end = idx_max > num_fano_bins * 0.80
    has_significant_drop = False
    if idx_max < num_fano_bins - 1: 
        drop_window_start = idx_max + 1
        drop_window_end = min(drop_window_start + int(num_fano_bins * 0.3), num_fano_bins) 
        if drop_window_end > drop_window_start:
            mean_f_after_peak = np.mean(smoothed_f_values[drop_window_start:drop_window_end])
            if not np.isnan(mean_f_after_peak):
                 has_significant_drop = mean_f_after_peak < (max_f * 0.90) 

    has_significant_rise = False
    if idx_max > 0: 
        rise_window_end = idx_max
        rise_window_start = max(0, idx_max - int(num_fano_bins * 0.3)) 
        if rise_window_end > rise_window_start:
            mean_f_before_peak = np.mean(smoothed_f_values[rise_window_start:rise_window_end])
            if not np.isnan(mean_f_before_peak):
                 has_significant_rise = mean_f_before_peak < (max_f * 0.90) 
                 
    if has_significant_drop and has_significant_rise and not is_peak_at_end and is_peak_central:
        return 'step'
    else:
        return 'ramp'


def evaluate_classifiers():
    scenarios = [
        {'name': 'Ramp Detection', 'model_type': 'ramp'},
        {'name': 'Step Detection', 'model_type': 'step'},
    ]

    for scenario in scenarios:
        correct_fano = 0
        correct_psth = 0

        for i in range(N_DATASETS_PER_CASE):
            current_params = generate_model_parameters(scenario['model_type'], T_DURATION_MS)
            
            if scenario['model_type'] == 'ramp':
                model = models.RampModel(beta=current_params['beta'], sigma=current_params['sigma'])
            else: 
                model = models.StepModel(m=current_params['m'], r=current_params['r'], x0=current_params['x0'])
            
            spikes, _, _ = model.simulate(Ntrials=N_TRIALS_CLASSIFY, T=T_DURATION_MS)
            
            predicted_type_fano = classify_model_by_fano(spikes, T_DURATION_MS, FANO_BIN_WIDTH_MS)
            if predicted_type_fano == scenario['model_type']:
                correct_fano += 1
            
            predicted_type_psth = classify_model_by_psth(spikes, T_DURATION_MS, PSTH_BIN_WIDTH_MS, args.show)
            if predicted_type_psth == scenario['model_type']:
                correct_psth += 1
            
        accuracy_fano = (correct_fano / N_DATASETS_PER_CASE) * 100
        accuracy_psth = (correct_psth / N_DATASETS_PER_CASE) * 100
        
        print()
        print(f"Result for {scenario['name']}:")
        print(f"  Fano Classifier: Correctly classified {correct_fano}/{N_DATASETS_PER_CASE} ({accuracy_fano:.2f}%)")
        print(f"  PSTH Classifier: Correctly classified {correct_psth}/{N_DATASETS_PER_CASE} ({accuracy_psth:.2f}%)")
        

if __name__ == "__main__":
    evaluate_classifiers() 