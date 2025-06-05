import numpy as np
import models
import scipy
import inference
from models_hmm import RampModelHMM, StepModelHMM
import matplotlib.pyplot as plt
from collections import OrderedDict as OD


np.set_printoptions(legacy='1.25') # don't show np.float; helps with debug

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

        Tmat = ramp._calculate_transition_matrix(T)
        pi = ramp._calculate_initial_distribution(T)

        state_rates = np.linspace(0, 1, K) * (Rh / T)

        LLH = inference.poisson_logpdf(data, state_rates)
        LLH = np.sum(LLH, axis=0)

        norm = inference.hmm_normalizer(pi, Tmat, LLH)
        return norm

    vllh = np.vectorize(lambda params: ramp_LLH_single(data, params))
    probs_grid = vllh(params)

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


def uniform_posterior(params_grid, log=True):
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

def bayes_factor(llh_grid, prior_grid, log=True):
    if log:
        return scipy.special.logsumexp(llh_grid + prior_grid)

def unnorm_posterior(llh_grid, prior_grid, log=True):
    if log:
        return llh_grid + prior_grid

def norm_posterior(llh_grid, prior_grid, log=True):
    if log:
        return llh_grid + prior_grid - bayes_factor(llh_grid, prior_grid, log=log)

def expectation(probs_grid, params_grid, log=True):
    if log:
        probs_grid = np.exp(probs_grid)

    param_names = list(params_grid.flat[0].keys())

    dtype = [(name, float) for name in param_names]
    param_values = np.zeros(params_grid.shape, dtype=dtype)

    for param in param_names:
        param_values[param] = np.vectorize(lambda x: x[param])(params_grid)

    expectations = {param: np.sum(param_values[param] * probs_grid) for param in param_names}

    return expectations


if __name__ == "__main__":

    T = 100
    Rh = 50

    specs = OD([('beta', [0, 1, 10]),
                ('sigma', [0, 0.5, 10]),
                ('T', T),
                ('Rh', Rh)])

    params_grid = make_params_grid(specs)

    true_beta = 1.3
    true_sigma = 0.2

    data, _, _ = RampModelHMM(beta=true_beta, sigma=true_sigma, Rh=Rh).simulate(Ntrials=100, T=T)

    LLH_probs_grid = ramp_LLH(data, params_grid)
    prior_probs_grid = uniform_posterior(params_grid)

    npost = norm_posterior(LLH_probs_grid, prior_probs_grid)
    print(expectation(npost, params_grid))

    print('bz')
