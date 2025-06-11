import numpy as np
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
        for bin_j in range(num_bins):
            start_time = bin_j * bin_width_ms
            end_time = (bin_j + 1) * bin_width_ms
            binned_sums[bin_j] += np.sum(spike_trains[trial_idx, start_time:end_time])
    
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