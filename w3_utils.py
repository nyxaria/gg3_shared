import numpy as np
import models
import scipy
import inference
from models_hmm import RampModelHMM, StepModelHMM
import matplotlib.pyplot as plt
from collections import OrderedDict as OD


np.set_printoptions(legacy='1.21') # don't show np.float; helps with debug

def ramp_LLH(data, params):
    # construct data
    defaults = {
        'K': 50,
        'Rh': 50,
        'x0': 0.2,
        'beta': 0.5,
        'sigma': 0.2,
        'T': 100,
        'filter': False
    }

    if type(params) == dict:
        params = np.array(params, dtype=object)

    def ramp_LLH_single(data, _params):

        _params = {**defaults, **_params}

        beta = _params['beta']
        sigma = _params['sigma']
        x0 = _params['x0']
        T = _params['T']
        K = _params['K']
        Rh = _params['Rh']

        ramp = RampModelHMM(beta, sigma, x0, K, Rh)
        # Calculate transition probability matrix between states
        Tmat = ramp._calculate_transition_matrix(T)
        
        # Calculate initial state distribution
        pi = ramp._calculate_initial_distribution(T)

        # Create array of firing rates for each state
        # Maps states (0 to 1) to rates (0 to Rh/T spikes per time bin)
        state_rates = np.linspace(0, 1, K) * (Rh / T)

        # Calculate log likelihood of observing spike counts given each state's rate
        # Returns matrix of log likelihoods for each timepoint and state
        LLH = inference.poisson_logpdf(data, state_rates)
        # summing here is wrong
        # LLH = np.sum(LLH, axis=0)

        # Calculate normalizing constant (log probability of the data)
        # using HMM forward algorithm
        norm = 0
        for trial_llh in LLH:
            trial_norm = inference.hmm_normalizer(pi, Tmat, trial_llh)
            norm += trial_norm

            # trial_norm ranges from ~-300 (off) to ~-200 (correct)

            # plt.matshow(np.exp(trial_llh))
            # plt.show()

        return norm

    probs_grid = np.empty_like(params, dtype=float)
    for idx, p in np.ndenumerate(params):
        probs_grid[idx] = ramp_LLH_single(data, p)

    # vllh = np.vectorize(lambda params: step_LLH_single(data, params))
    # probs_grid = vllh(params)



    return probs_grid


def step_LLH(data, params):
    # construct data
    defaults = {
        'K': 50,
        'Rh': 50,
        'x0': 0.2,
        'm': 50,
        'r': 10,
        'T': 100,
    }

    if type(params) == dict:
        params = np.array(params, dtype=object)

    def step_LLH_single(data, _params):

        _params = {**defaults, **_params}

        m = _params['m']
        if not _params['r'].is_integer():
            print('rounding down floating point r:', params['r'])

        r = int(_params['r'])
        x0 = _params['x0']
        Rh = _params['Rh']
        T = _params['T']

        K = 50


        step = StepModelHMM(m=m, r=r, x0=x0, Rh=Rh)

        Tmat = step._calculate_transition_matrix_exact(T=T)
        pi = step._calculate_initial_distribution_exact()

        # compensate by changing pi
        pi = pi.T @ np.linalg.matrix_power(Tmat, r)
        pi = pi.T

        state_rates = np.ones(int(r) + 1) * (x0 * Rh) / T
        state_rates[-1] = Rh / T

        LLH = inference.poisson_logpdf(data, state_rates)
        # LLH = np.sum(LLH, axis=0)

        norm = 0
        for trial_llh in LLH:
            trial_norm = inference.hmm_normalizer(pi, Tmat, trial_llh)
            norm += trial_norm

        return norm

    probs_grid = np.empty_like(params, dtype=float)
    for idx, p in np.ndenumerate(params):
        probs_grid[idx] = step_LLH_single(data, p)

    # vllh = np.vectorize(lambda params: step_LLH_single(data, params))
    # probs_grid = vllh(params)

    return probs_grid

def make_params_grid(specs):
    """
    Create a grid of all possible parameter combinations, where each cell is a tuple of parameters.

    Parameters:
    -----------
    specs : Ordered dict
        A dictionary where keys are parameter names and values are either:
        - A tuple `(start, stop, num)` for `np.linspace`, or
        - A list/array of explicit values.

    Returns:
    --------
    list of list of tuples
        A grid (2D list) where each cell contains a tuple of parameter values.
    dict
        A dictionary of parameter names and their generated values.
    """
    # Generate parameter values
    param_values = {}
    for param, spec in specs.items():
        if isinstance(spec, (tuple, list)) and len(spec) == 3:
            param_values[param] = np.linspace(*spec)
        else:
            param_values[param] = np.asarray(spec)

        if isinstance(spec, (float, int)):
            param_values[param] = np.array([spec])

    mesh = np.meshgrid(*param_values.values(), indexing='ij')

    grid_shape = mesh[0].shape
    param_names = list(param_values.keys())
    param_grid = np.empty(grid_shape, dtype=object)

    for indices in np.ndindex(*grid_shape):
        current_params = {list(specs.keys())[i]: mesh[i][indices] for i in range(len(param_names))}
        param_grid[indices] = current_params

    return np.squeeze(param_grid)


def get_param_values(params_grid, param_name):
    # get some named param from params grid, for testing
    return np.vectorize(lambda x: x[param_name])(params_grid)


def uniform_prior(params_grid, log=True):
    """
    Generate a uniform posterior probability grid with the same shape as params_grid.

    Parameters:
    - params_grid: numpy array of any shape
    - log: boolean, if True returns log probabilities, otherwise regular probabilities

    Returns:
    - probs_grid: numpy array with same shape as params_grid containing uniform probabilities
                  that sum to 1 (or log probabilities that sum to 1 when exponentiated)
    """
    probs_grid = np.ones_like(params_grid, dtype=float)

    probs_grid = probs_grid / probs_grid.sum()

    if log:
        probs_grid = np.log(probs_grid)

    return probs_grid


def gaussian_prior(params_grid, mu, cov, log=True):
    """
    Generate a gaussian posterior probability grid with the same shape as params_grid.

    Parameters:
    - params_grid: numpy array of any shape containing parameter dictionaries
    - mu: dict with mean values for each parameter
    - cov: 2D array or dict of parameter variances/covariances
    - log: boolean, if True returns log probabilities, otherwise regular probabilities

    Returns:
    - probs_grid: numpy array with same shape as params_grid containing probabilities
                  that sum to 1 (or log probabilities that sum to 1 when exponentiated)
    """
    shape = params_grid.shape
    sorted_mu_keys = sorted(mu.keys())
    param_names = sorted(list(params_grid.flat[0].keys()))

    varied_params = []
    for i, param in enumerate(param_names):
        if param in mu:
            varied_params.append(get_param_values(params_grid, param))

    # Convert mu dict to vector in same order as param_names
    mu_vec = np.array([mu[param] for param in sorted_mu_keys])

    # Prepare covariance matrix if provided as dict
    # also, i realised that this wouldve been cleaner if cov matrix only was allowed
    # but since the param grid dict is not ordered we have no way of knowing
    # how the order of the cov matrix corresponds, and I don't want to refactor for now

    if isinstance(cov, dict):
        cov_matrix = np.zeros((len(varied_params), len(varied_params)))
        for i, pi in enumerate(sorted_mu_keys):
            for j, pj in enumerate(sorted_mu_keys):
                key = (pi, pj) if (pi, pj) in cov else (pj, pi)
                if key not in cov:
                    if pi == pj:
                        raise ValueError(f"Covariance matrix is missing key {key}")
                    else:
                        cov[key] = 0 # assume uncorrelated

                cov_matrix[i, j] = cov[key]

    else:
        cov_matrix = cov

    # Calculate Gaussian probability density
    varied_params = np.array(varied_params)
    reshaped_params = np.moveaxis(varied_params, 0, -1).reshape(-1, len(mu_vec))

    diff = reshaped_params - mu_vec

    # exp_term =  -0.5 * np.einsum('ij, jk, ik->i', diff, np.linalg.inv(cov_matrix), diff)
    exp_term = -0.5 * np.sum((diff @ np.linalg.inv(cov_matrix)) * diff, axis=1)

    exp_term = exp_term.reshape(shape)

    probs_grid = np.exp(exp_term)
    probs_grid /= np.sum(probs_grid)

    if log:
        probs_grid = np.log(probs_grid)

    return probs_grid


def sample_from_grid(probs_grid, params_grid=None, log=True):
    if log:
        probs_grid = np.exp(probs_grid)

    flat = probs_grid.flatten()
    flat_idx = np.random.choice(len(flat), p=flat) # sample with probs grid as our dist

    idx = np.unravel_index(flat_idx, probs_grid.shape)

    if params_grid is not None:
        return params_grid[idx]

    else:
        return idx

def marginal_likelihood(llh_grid, prior_grid, log=True):
    if log:
        return scipy.special.logsumexp(llh_grid + prior_grid)

def unnorm_posterior(llh_grid, prior_grid, log=True):
    if log:
        return llh_grid + prior_grid

def norm_posterior(llh_grid, prior_grid, log=True):
    if log:
        return llh_grid + prior_grid - marginal_likelihood(llh_grid, prior_grid, log=log)

def expectation(probs_grid, params_grid, log=True):
    if log:
        probs_grid = np.exp(probs_grid)

    param_names = list(params_grid.flat[0].keys())

    dtype = [(name, float) for name in param_names]
    param_values = np.zeros(params_grid.shape, dtype=dtype)

    for param in param_names:
        param_values[param] = get_param_values(params_grid, param)

    expectations = {param: np.sum(param_values[param] * probs_grid) for param in param_names}

    return expectations


def posterior_std_dev(probs_grid, params_grid, posterior_means, log=True):
    """
    Calculate the posterior standard deviation for each parameter.

    Parameters:
    - probs_grid: numpy array, posterior probabilities (can be log probabilities).
    - params_grid: numpy array, grid of parameter dictionaries.
    - posterior_means: dict, pre-calculated posterior means for each parameter.
    - log: boolean, if True, probs_grid is in log scale.

    Returns:
    - std_devs: dict, posterior standard deviation for each parameter.
    """
    if log:
        max_log_prob = np.max(probs_grid)
        probs_grid_linear = np.exp(probs_grid - max_log_prob)
        probs_grid_linear = probs_grid_linear / np.sum(probs_grid_linear)
    else:
        probs_grid_linear = probs_grid / np.sum(probs_grid)


    param_names = list(params_grid.flat[0].keys())
    std_devs = {}

    for param in param_names:
        if param not in posterior_means: # Skip if param not in posterior_means (e.g. T, Rh, K)
            if param in params_grid.flat[0] and isinstance(params_grid.flat[0][param], (int, float)):
                 pass
            else:
                continue


        # E[X^2]
        try:
            # Vectorized extraction of param**2
            squared_values = np.array([d[param]**2 for d in params_grid.flat]).reshape(params_grid.shape)
        except TypeError:
            continue


        expected_sq_value = np.sum(squared_values * probs_grid_linear)
        
        # Variance = E[X^2] - (E[X])^2
        variance = expected_sq_value - (posterior_means[param]**2)
        
        if variance < 0:
            variance = 0.0
            
        std_devs[param] = np.sqrt(variance)
        
    return std_devs


if __name__ == "__main__":

    T = 100
    Rh = 50
    N_TRIALS = 10

    specs = OD([('m', [25, 75, 15]),
                ('r', [1, 6, 6]),
                ('T', T),
                ('Rh', Rh)])

    params_grid = make_params_grid(specs)

    step_true_params = {'m': 50, 'r': 2, 'x0': 0.2}


    data, _, _ = StepModelHMM(m=step_true_params['m'], r=step_true_params['r'], Rh=Rh).simulate_exact(Ntrials=N_TRIALS, T=T, delay_compensation=True)

    LLH_probs_grid = step_LLH(data, params_grid)
    prior_probs_grid = uniform_prior(params_grid)

    npost = norm_posterior(LLH_probs_grid, prior_probs_grid)
    plt.matshow(np.exp(npost))
    plt.show()


    print(step_true_params)
    print(expectation(npost, params_grid))

    get_param_values(params_grid, 'm')
    get_param_values(params_grid, 'r')


    '''specs = OD([('beta', [0, 1, 10]),
                ('sigma', [0, 0.5, 10]),
                ('T', T),
                ('Rh', Rh)])'''

    '''K = 25
    T_MS = 100
    RH = 500
    M_GRID = 7

    true_params = {'beta': 1.0, 'sigma': 0.2, 'x0': 0.2}

    specs = OD([
        ('beta', np.linspace(0, 4, M_GRID)),
        ('sigma', np.exp(np.linspace(np.log(0.04), np.log(4), M_GRID))),
        ('x0', true_params['x0']),
        ('K', K),
        ('T', T_MS),
        ('Rh', RH)
    ])

    params_grid = make_params_grid(specs)



    data, _, _ = RampModelHMM(beta=true_params['beta'], sigma=true_params['sigma'], Rh=RH).simulate(Ntrials=1, T=T_MS)

    LLH_probs_grid = ramp_LLH(data, params_grid)
    prior_probs_grid = uniform_prior(params_grid)

    npost = norm_posterior(LLH_probs_grid, prior_probs_grid)

    plt.matshow(np.exp(npost))
    plt.show()

    print(expectation(npost, params_grid))'''
