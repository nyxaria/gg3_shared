import numpy as np
import os
import argparse
import concurrent.futures
from tqdm import tqdm
from collections import OrderedDict as OD
import scipy.special
from scipy.stats import norm, truncnorm
from scipy.optimize import curve_fit
import pickle
from pathlib import Path

from models_hmm import RampModelHMM, StepModelHMM
import w3_utils
from utils import calculate_psth, classify_model_by_psth

# --- Core Logic for Task 3.2.3 ---

def calculate_log_prior_grid(param_grid, param_ranges, sigma_frac):
    log_prior_grid = np.zeros_like(param_grid, dtype=float)
    for idx, params in np.ndenumerate(param_grid):
        log_prob = 0
        for name, (min_p, max_p) in param_ranges.items():
            if name in params:
                mean_p, std_p = (min_p + max_p) / 2, (max_p - min_p) * sigma_frac
                if std_p > 0:
                    log_prob += norm.logpdf(params[name], loc=mean_p, scale=std_p)
        log_prior_grid[idx] = log_prob
    return log_prior_grid - scipy.special.logsumexp(log_prior_grid)

def _comparison_worker(args_tuple):
    (true_model_type, prior_type, ramp_ranges, step_ranges, sigma_frac, 
     ramp_grid, step_grid, ramp_log_prior, step_log_prior,
     K, T, Rh, x0, n_trials, bin_w) = args_tuple

    if prior_type == 'gaussian':
        if true_model_type == 'ramp':
            params = {n: truncnorm.rvs((min_p - (min_p+max_p)/2)/((max_p-min_p)*sigma_frac), (max_p - (min_p+max_p)/2)/((max_p-min_p)*sigma_frac), loc=(min_p+max_p)/2, scale=(max_p-min_p)*sigma_frac) for n, (min_p, max_p) in ramp_ranges.items()}
            model = RampModelHMM(**params, x0=x0, K=K, Rh=Rh)
            data, _, _ = model.simulate(Ntrials=n_trials, T=T)
        else: # step
            params = {n: truncnorm.rvs((min_p - (min_p+max_p)/2)/((max_p-min_p)*sigma_frac), (max_p - (min_p+max_p)/2)/((max_p-min_p)*sigma_frac), loc=(min_p+max_p)/2, scale=(max_p-min_p)*sigma_frac) for n, (min_p, max_p) in step_ranges.items() if n != 'r'}
            params['r'] = np.random.choice(np.arange(step_ranges['r'][0], step_ranges['r'][1]+1))
            model = StepModelHMM(**params, x0=x0, Rh=Rh)
            data, _, _ = model.simulate_exact(Ntrials=n_trials, T=T, delay_compensation=True)
    elif prior_type == 'uniform':
        if true_model_type == 'ramp':
            params = {n: np.random.uniform(min_p, max_p) for n, (min_p, max_p) in ramp_ranges.items()}
            model = RampModelHMM(**params, x0=x0, K=K, Rh=Rh)
            data, _, _ = model.simulate(Ntrials=n_trials, T=T)
        else: # step
            params = {n: np.random.uniform(min_p, max_p) for n, (min_p, max_p) in step_ranges.items() if n != 'r'}
            params['r'] = np.random.randint(step_ranges['r'][0], step_ranges['r'][1]+1)
            model = StepModelHMM(**params, x0=x0, Rh=Rh)
            data, _, _ = model.simulate_exact(Ntrials=n_trials, T=T, delay_compensation=True)

    ramp_llh = w3_utils.ramp_LLH(data, ramp_grid)
    step_llh = w3_utils.step_LLH(data, step_grid)
    ramp_marginal = scipy.special.logsumexp(ramp_llh + ramp_log_prior)
    step_marginal = scipy.special.logsumexp(step_llh + step_log_prior)
    bayes_correct = (1 if ramp_marginal > step_marginal else 0) if true_model_type == 'ramp' else (1 if step_marginal > ramp_marginal else 0)

    adhoc_pred = classify_model_by_psth(data, T, bin_w)
    adhoc_correct = 1 if adhoc_pred == true_model_type else 0
    return bayes_correct, adhoc_correct

def main():
    parser = argparse.ArgumentParser(description="Task 3.2.3: Compare Bayesian and Ad-hoc Classifiers")
    parser.add_argument('--n_datasets', type=int, default=100)
    parser.add_argument('--n_trials', type=int, default=300)
    parser.add_argument('--m_grid', type=int, default=10)
    parser.add_argument('--prior', type=str, default='gaussian', choices=['gaussian', 'uniform'], help="Prior to use for data generation and inference.")
    parser.add_argument('--no_cache', action='store_true', help="Disable caching and force re-computation.")
    args = parser.parse_args()

    K, T_MS, RH, X0, SIGMA_FRAC, PSTH_BIN_W = 50, 100, 50, 0.2, 0.25, 25
    
    ramp_ranges = OD([('beta', (0, 4)), ('sigma', (0.04, 4))])
    step_ranges = OD([('m', (0, T_MS * 3/4)), ('r', (1, 6))])

    ramp_specs = OD(list(ramp_ranges.items()) + [('x0', X0), ('K', K), ('T', T_MS), ('Rh', RH)])
    step_specs = OD(list(step_ranges.items()) + [('x0', X0), ('T', T_MS), ('Rh', RH)])
    for k in ['m', 'r']:
        step_specs[k] = np.arange(step_ranges[k][0], step_ranges[k][1]+1) if k == 'r' else np.linspace(*step_ranges[k], args.m_grid)
    ramp_specs['sigma'] = np.linspace(*ramp_ranges['sigma'], args.m_grid)
    ramp_specs['beta'] = np.linspace(*ramp_ranges['beta'], args.m_grid)

    ramp_grid = w3_utils.make_params_grid(ramp_specs)
    step_grid = w3_utils.make_params_grid(step_specs)
    
    if args.prior == 'gaussian':
        ramp_log_prior = calculate_log_prior_grid(ramp_grid, ramp_ranges, SIGMA_FRAC)
        step_log_prior = calculate_log_prior_grid(step_grid, step_ranges, SIGMA_FRAC)
    else: # uniform
        ramp_log_prior = w3_utils.uniform_prior(ramp_grid)
        step_log_prior = w3_utils.uniform_prior(step_grid)

    cache_dir = Path("cache")
    cache_dir.mkdir(exist_ok=True)

    print(f"--- Comparing Classifiers | Prior: {args.prior.upper()} | N_datasets={args.n_datasets}, N_trials={args.n_trials}, Sigma_frac={SIGMA_FRAC} ---")
    totals = {'bayes': {'ramp': 0, 'step': 0}, 'adhoc': {'ramp': 0, 'step': 0}}
    
    for model_type in ['ramp', 'step']:
        print(f"\nGenerating data from {model_type.capitalize()} model...")
        
        cache_filename = f"task3.2.3_{model_type}_p-{args.prior}_N{args.n_datasets}_T{args.n_trials}_M{args.m_grid}_S{SIGMA_FRAC}.pkl"
        cache_path = cache_dir / cache_filename
        
        results = []
        if cache_path.exists() and not args.no_cache:
            try:
                with open(cache_path, 'rb') as f:
                    results = pickle.load(f)
                print(f"Loaded {len(results)} cached results from {cache_path}")
            except (pickle.UnpicklingError, EOFError):
                 print(f"Cache file {cache_path} is corrupted. Starting fresh.")
                 results = []

        n_remaining = args.n_datasets - len(results)

        if n_remaining <= 0:
            print(f"Found {len(results)} cached results, which is sufficient. Skipping computation.")
        else:
            print(f"Found {len(results)} cached results. Running {n_remaining} more simulations.")
            tasks = [(model_type, args.prior, ramp_ranges, step_ranges, SIGMA_FRAC, 
                      ramp_grid, step_grid, ramp_log_prior, step_log_prior,
                      K, T_MS, RH, X0, args.n_trials, PSTH_BIN_W) for _ in range(n_remaining)]
            
            with concurrent.futures.ProcessPoolExecutor() as executor:
                new_results_iterator = executor.map(_comparison_worker, tasks)
                
                for res in tqdm(new_results_iterator, total=n_remaining, desc=f"Running {model_type}"):
                    results.append(res)
                    with open(cache_path, 'wb') as f:
                        pickle.dump(results, f)

        if results:
            bayes_correct, adhoc_correct = zip(*results)
            totals['bayes'][model_type] = np.sum(bayes_correct)
            totals['adhoc'][model_type] = np.sum(adhoc_correct)

    print("\n" + "="*20 + " RESULTS " + "="*20)
    for name, t in totals.items():
        ramp_acc = (t['ramp'] / args.n_datasets * 100) if args.n_datasets > 0 else 0
        step_acc = (t['step'] / args.n_datasets * 100) if args.n_datasets > 0 else 0
        total_acc = (ramp_acc + step_acc) / 2
        print(f"\n{name.capitalize()} Classifier:")
        print(f"  Accuracy on Ramp data: {ramp_acc:.1f}%")
        print(f"  Accuracy on Step data: {step_acc:.1f}%")
        print(f"  {'':-^30}\n  Overall Accuracy:      {total_acc:.1f}%")
    print("="*49)

if __name__ == '__main__':
    main() 