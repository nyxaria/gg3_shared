import numpy as np
import matplotlib.pyplot as plt
import models
import argparse
from scipy.optimize import curve_fit
import traceback
from utils import calculate_psth, calculate_fano_factor, generate_model_parameters
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

def sigmoid_func(t, L, k, t0, y0):
    """ 
    L: Amplitude of the step (max val - min val)
    k: Steepness of the sigmoid
    t0: Midpoint of the sigmoid
    y0: y(t=0)

    y(t) = y0 + L / (1 + exp(-k * (t - t0)))
    """
    return y0 + L / (1 + np.exp(-k * (t - t0)))


def ramp_func(t, m, t1, A, b):
    """
    m: Slope of the initial linear segment
    t1: Transition time from linear to exponential
    A: Absolute asymptotic value the curve settles at
    b: y(t=0)

    y(t) = b + m*t                                   for t <= t1
    y(t) = A - (A - y(t1)) * exp(-beta * (t - t1))   for t > t1
      beta = m / (A - y(t1)) to ensure slope continuity
    """
    y = np.zeros_like(t, dtype=float)
    
    # linear line (t < t1)
    y_val_at_t1 = b + m * t1

    linear_indices = np.where(t < t1)[0]
    y[linear_indices] = b + m * t[linear_indices]

    # exponential curve (t > t1)
    exp_indices = np.where(t >= t1)[0]
    if exp_indices.size == 0: # transition is at end ?
        return y

    # if the asymptote is very close to the value at t1, return the asymptote
    if abs(A - y_val_at_t1) < 1e-9: 
        y[exp_indices] = A 
        return y
        
    # if the asymptote is below the value at t1, return a large penalty (we should never go down in our psth)
    if A < y_val_at_t1:
        y[exp_indices] = 1e12
        return y

    beta = m / (A - y_val_at_t1)
    y[exp_indices] = A - (A - y_val_at_t1) * np.exp(-beta * (t[exp_indices] - t1))
    
    return y

def fit_sigmoid(x_data, y_data):
    min_y, max_y = (np.min(y_data), np.max(y_data)) if y_data.size > 0 else (0,1)
    guess_sigmoid = [max_y - min_y, 5.0, 0.5, min_y]
    
    bounds_sigmoid = (
        [0, 1e-3, 0, 0],  
        [max_y*2 +1 if max_y >0 else 2, 100.0, 1.0, max_y+1 if max_y >0 else 1]
    )
    
    guess_sigmoid = [np.clip(guess_sigmoid[i], bounds_sigmoid[0][i], bounds_sigmoid[1][i]) for i in range(4)]

    params_sigmoid, _ = curve_fit(sigmoid_func, x_data, y_data, p0=guess_sigmoid, bounds=bounds_sigmoid, maxfev=10000)
    return params_sigmoid

def fit_ramp(x_data, y_data):
    x_data = np.asarray(x_data, dtype=float)
    y_data = np.asarray(y_data, dtype=float)

    b_guess = y_data[0]
    m_guess = (y_data[1] - y_data[0]) / (x_data[1] - x_data[0])
    A_guess = y_data[-1]
    t1_guess = x_data[-1]/2

    y_at_t1_guess = b_guess + m_guess * t1_guess
    if A_guess <= y_at_t1_guess and m_guess * t1_guess > -b_guess : # check if ramp part is positive
        A_guess = y_at_t1_guess * 1.2 if y_at_t1_guess > 0 else y_at_t1_guess + abs(y_at_t1_guess*0.2) + 1.0


    initial_params = [m_guess, t1_guess, A_guess, b_guess]

    # bounds
    max_y_val = np.max(y_data)
    min_y_val = np.min(y_data)

    bounds = [
        [0.1, 1000],  # m
        [x_data[0], x_data[-1]],  # t1
        [min_y_val * 0.5 if min_y_val > 0 else -max_y_val, max_y_val * 4.0 + 1.0 if max_y_val > 0 else 1000.0],  # A
        [0.0, max_y_val * 1.5 + 1.0]  # b
    ]

    initial_params = [np.clip(initial_params[i], bounds[i][0], bounds[i][1]) for i in range(4)]

    # curve fit expects a different format *sigh*
    lower_bounds = [b[0] for b in bounds]
    upper_bounds = [b[1] for b in bounds]
    
    popt, _ = curve_fit(
        ramp_func,
        x_data,
        y_data,
        p0=initial_params,
        bounds=(lower_bounds, upper_bounds),
        maxfev=10000, 
    )
    
    m_opt, t1_opt, A_opt, b_opt = popt
    
    return m_opt, t1_opt, A_opt, b_opt

def classify_model_by_psth(spike_trains, t_duration_ms, psth_bin_width_ms, show_plot):
    time_bins, psth_hz = calculate_psth(spike_trains, t_duration_ms, psth_bin_width_ms)

    t_data_normalized = time_bins / t_duration_ms
    y_data = psth_hz
    y_data = y_data[~np.isnan(y_data)]

    if y_data.size == 0:
        print("Warning: y_data is empty after NaN removal in classify_model_by_psth. Returning random choice.")
        return np.random.choice(['ramp', 'step'])

    # fit to sigmoid curve
    ssr_sigmoid = 100000

    try:
        # initial guess for sigmoid
        L_sig, k_sig, t0_sig, b_sig_val = fit_sigmoid(t_data_normalized, y_data)
        y_fit_sigmoid = sigmoid_func(t_data_normalized, L_sig, k_sig, t0_sig, b_sig_val)
        ssr_sigmoid = np.sum((y_data - y_fit_sigmoid)**2)
        if show_plot:
            print(f"Sigmoid fit: ssr={ssr_sigmoid:6.2f}, L={L_sig:6.2f}, k={k_sig:6.2f}, t0={t0_sig:6.2f}, b={b_sig_val:6.2f}")

    except Exception as e:
        print(traceback.format_exc())
    
    ssr_ramp = 100000

    try:
        m_fit, t1_fit, A_fit, b_ramp_fit = fit_ramp(t_data_normalized, y_data) 
        y_fit_ramp = ramp_func(t_data_normalized, m_fit, t1_fit, A_fit, b_ramp_fit)
        ssr_ramp = np.sum((y_data - y_fit_ramp)**2)
        if show_plot:
            print(f"Ramp fit: ssr={ssr_ramp:.3f}, m={m_fit:.3f}, t1={t1_fit:.3f}, A={A_fit:.3f}, b={b_ramp_fit:.3f}")
    except Exception as e:
        print(traceback.format_exc())
        

    # if ssr_ramp < ssr_sigmoid + 10 or (t0_sig < 0.1 or t0_sig > 0.9 or b_sig_val < 0.5):
    if ssr_ramp < ssr_sigmoid + 10:
        final_classification = 'ramp'
    else:
        final_classification = 'step'


    if show_plot:
        plt.figure(figsize=(10, 6))
        plt.plot(t_data_normalized, y_data, 'ko', label='Original PSTH', markersize=4)
        plt.plot(t_data_normalized, y_fit_sigmoid, 'r--', label=f'Sigmoid Fit (SSR: {ssr_sigmoid:.2f})')
        plt.plot(t_data_normalized, y_fit_ramp, 'g--', label=f'Ramp Fit (SSR: {ssr_ramp:.2f})')
        plt.xlabel("Normalized Time")
        plt.ylabel("PSTH (Hz)")
        plt.title(f"PSTH Fit Classification: {final_classification.upper()}")
        plt.legend()
        plt.grid(True)
        plt.show()

    return final_classification


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