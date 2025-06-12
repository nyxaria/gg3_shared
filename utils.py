import numpy as np
import scipy.special
from scipy.optimize import curve_fit

def calculate_psth(spike_trains, t_duration_ms, bin_width_ms):
    # bin = np.convolve(bin, np.ones(bin_width_ms)/bin_width_ms, mode='valid')
    # spike_times = np.linspace(0, t_duration_ms, num = bin.shape[0], endpoint = False)

    # return spike_times, bin * t_duration_ms

    n_trials = spike_trains.shape[0]
    num_bins = int(t_duration_ms / bin_width_ms)
    psth_time_bins_ms = np.arange(num_bins) * bin_width_ms
    binned_sums = np.zeros(num_bins)

    # sum the spikes across trials into bins
    for trial_idx in range(n_trials):
        for bin_idx in range(num_bins):
            start_time = bin_idx * bin_width_ms
            end_time = (bin_idx + 1) * bin_width_ms
            binned_sums[bin_idx] += np.sum(spike_trains[trial_idx, start_time:end_time])
    
    # average and convert to Hz
    mean_spikes_per_bin = binned_sums / n_trials
    psth_values_hz = mean_spikes_per_bin / (bin_width_ms / 1000.0)  # convert to Hz
    
    return psth_time_bins_ms, psth_values_hz

def calculate_fano_factor(spike_trains, t_duration_ms, bin_width_ms):
    n_trials = spike_trains.shape[0]
    num_bins_fano = int(t_duration_ms / bin_width_ms)
    fano_time_bins_ms = np.arange(num_bins_fano) * bin_width_ms
    binned_counts_per_trial = np.zeros((n_trials, num_bins_fano))

    for trial_idx in range(n_trials):
        for bin_idx in range(num_bins_fano):
            start_time = bin_idx * bin_width_ms
            end_time = (bin_idx + 1) * bin_width_ms
            binned_counts_per_trial[trial_idx, bin_idx] = np.sum(spike_trains[trial_idx, start_time:end_time])
            
    mean_counts_per_bin = np.mean(binned_counts_per_trial, axis=0)
    variance_counts_per_trial = np.var(binned_counts_per_trial, axis=0, ddof=0)
    
    fano_factors = np.full_like(mean_counts_per_bin, np.nan, dtype=float)
    
    # calculate Fano factor only where mean is non zero
    non_zero_mean_indices = mean_counts_per_bin != 0
    fano_factors[non_zero_mean_indices] = variance_counts_per_trial[non_zero_mean_indices] / mean_counts_per_bin[non_zero_mean_indices]
    
    return fano_time_bins_ms, fano_factors

def generate_model_parameters(model_type_to_generate, t_duration_ms):
    if model_type_to_generate == 'ramp':
        beta = np.random.uniform(0, 4.0)
        sigma = np.exp(np.random.uniform(np.log(0.04), np.log(4)))
        return {'beta': beta, 'sigma': sigma}
    else:
        m = np.random.uniform(t_duration_ms * 0.25, t_duration_ms * 0.75)
        r = np.random.uniform(0.5, 6.0)
        x0 = np.random.uniform(0, 0.5)
        return {'m': m, 'r': r, 'x0': x0}

def generate_random_model_parameters(t_duration_ms):
    model_type = np.random.choice(['ramp', 'step'])
    if model_type == 'ramp':
        beta = np.random.uniform(0.1, 50.0)
        sigma = np.random.uniform(0.1, 5.0)
        params = {'beta': beta, 'sigma': sigma}
    else:
        # Use the passed t_duration_ms for calculating 'm'
        m = np.random.uniform(t_duration_ms * 0.1, t_duration_ms * 0.9)
        r = np.random.uniform(0.1, 50.0)
        params = {'m': m, 'r': r}
    return params

# --- Ad-hoc Classifier Functions ---

def sigmoid_func(t, L, k, t0, y0):
    """ 
    L: Amplitude of the step (max val - min val)
    k: Steepness of the sigmoid
    t0: Midpoint of the sigmoid
    y0: y(t=0)

    y(t) = y0 + L / (1 + exp(-k * (t - t0)))
    """
    return y0 + L * scipy.special.expit(k * (t - t0))


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

def classify_model_by_psth(spike_trains, t_duration_ms, psth_bin_width_ms, show_plot=False):
    import matplotlib.pyplot as plt
    import traceback

    time_bins, psth_hz = calculate_psth(spike_trains, t_duration_ms, psth_bin_width_ms)

    t_data_normalized = time_bins / t_duration_ms
    y_data = psth_hz
    y_data = y_data[~np.isnan(y_data)]

    if y_data.size < 4:
        # print("Warning: y_data is empty after NaN removal in classify_model_by_psth. Returning random choice.")
        return np.random.choice(['ramp', 'step'])

    # fit to sigmoid curve
    ssr_sigmoid = float('inf')

    try:
        # initial guess for sigmoid
        L_sig, k_sig, t0_sig, b_sig_val = fit_sigmoid(t_data_normalized, y_data)
        y_fit_sigmoid = sigmoid_func(t_data_normalized, L_sig, k_sig, t0_sig, b_sig_val)
        ssr_sigmoid = np.sum((y_data - y_fit_sigmoid)**2)
        if show_plot:
            print(f"Sigmoid fit: ssr={ssr_sigmoid:6.2f}, L={L_sig:6.2f}, k={k_sig:6.2f}, t0={t0_sig:6.2f}, b={b_sig_val:6.2f}")

    except Exception:
        if show_plot:
            print(traceback.format_exc())
    
    ssr_ramp = float('inf')

    try:
        m_fit, t1_fit, A_fit, b_ramp_fit = fit_ramp(t_data_normalized, y_data) 
        y_fit_ramp = ramp_func(t_data_normalized, m_fit, t1_fit, A_fit, b_ramp_fit)
        ssr_ramp = np.sum((y_data - y_fit_ramp)**2)
        if show_plot:
            print(f"Ramp fit: ssr={ssr_ramp:.3f}, m={m_fit:.3f}, t1={t1_fit:.3f}, A={A_fit:.3f}, b={b_ramp_fit:.3f}")
    except Exception:
        if show_plot:
            print(traceback.format_exc())
        

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