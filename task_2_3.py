import numpy as np
import models
import scipy
import inference
from models_hmm import RampModelHMM, StepModelHMM
import matplotlib.pyplot as plt
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

K = 50 
Rh = 100
x0 = 0.2
beta = 0.5
T = 100
filter = True


np.set_printoptions(legacy='1.25') # don't show np.float; helps with debug


def ramp_HMM_inference(model_parameters=None, test_filtering=False):
    defaults = {
        'K': 50,
        'Rh': 100,
        'x0': 0.2,
        'beta': 0.5,
        'sigma': 0.2,
        'T': 100,
        'filter': False
    }

    if model_parameters is None:
        model_parameters = {}

    params = {**defaults, **model_parameters}

    # Extract parameters (could also just use params['K'], etc. in your code)
    K = params['K']
    Rh = params['Rh']
    x0 = params['x0']
    beta = params['beta']
    T = params['T']
    sigma = params['sigma']
    filter = params['filter']

    ramp = RampModelHMM(beta, sigma, x0, K, Rh)
    ramp_spikes, states, rates = ramp.simulate(Ntrials=1, T=T, return_state_indices=True)
    Tmat = ramp._calculate_transition_matrix(T)
    pi = ramp._calculate_initial_distribution(T)

    state_rates = np.linspace(0, 1, K) * (Rh / T)

    LLH = inference.poisson_logpdf(ramp_spikes, state_rates)
    LLH = np.sum(LLH, axis=0)

    ex, norm = inference.hmm_expected_states(pi, Tmat, LLH, filter=False)
    expected_s = ex @ np.arange(K)  # this is expected s_t

    if test_filtering:
        fex, fnorm = inference.hmm_expected_states(pi, Tmat, LLH, filter=True)
        fexpected_s = fex @ np.arange(K)  # this is expected s_t
        return ex, fex, expected_s, fexpected_s, states.flatten()

    return ex, expected_s, states.flatten()


def step_HMM_inference(model_parameters=None, test_filtering=False, compress_states=False):
    defaults = {
        'K': 50,
        'Rh': 100,
        'x0': 0.2,
        'm': 50,
        'r': 10,
        'T': 100,
    }

    if model_parameters is None:
        model_parameters = {}

    params = {**defaults, **model_parameters}

    # Extract parameters (could also just use params['K'], etc. in your code)
    # K = params['K']
    m = params['m']
    r = params['r']
    x0 = params['x0']
    Rh = params['Rh']
    T = params['T']

    step = StepModelHMM(m=m, r=r, x0=x0, Rh=Rh)

    Tmat = step._calculate_transition_matrix_exact()
    pi = step._calculate_initial_distribution_exact()

    step_spikes, jumps, states = step.simulate_exact(T=T, return_state_indices=True)

    state_rates = np.ones(int(r) + 1) * (x0 * Rh) / T
    state_rates[-1] = Rh/T

    LLH = inference.poisson_logpdf(step_spikes, state_rates)
    LLH = np.sum(LLH, axis=0)

    ex, norm = inference.hmm_expected_states(pi, Tmat, LLH, filter=False)

    if compress_states: # compress all non-up states into one state, so ex will always be 2.
        ex = np.hstack((
            np.sum(ex[:, :-1], axis=1, keepdims=True),
            ex[:, -1:]
        ))

        states = np.hstack((
            np.sum(states[:, :-1], axis=1, keepdims=True),
            states[:, -1:]
        ))

    posterior_rate_binary = (ex[:, -1] >= 0.5).astype(int)

    if test_filtering:
        fex, fnorm = inference.hmm_expected_states(pi, Tmat, LLH, filter=True)
        fposterior_rate_binary = (fex[:, -1] >= 0.5).astype(int)

        if compress_states:
            fex = np.hstack((
                np.sum(fex[:, :-1], axis=1, keepdims=True),
                fex[:, -1:]
            ))

        return ex, fex, posterior_rate_binary, fposterior_rate_binary, states.flatten()
    return ex, posterior_rate_binary, states.flatten()

def compress_states(arr):
    # sum the first n-1 cols along axis 1, keep the last col the same. shape (n, m) -> (n, 2)
    # if 1d array is passed we assume n=1

    if arr.ndim == 1:
        arr = np.array([arr])

    return np.hstack((
                np.sum(arr[:, :-1], axis=1, keepdims=True),
                arr[:, -1:]
            ))

def cross_entropy(ex, true_s, base=2, time_average=False, time_sum=False):
    assert ex.shape[0] == len(true_s)

    # delta_indices = np.column_stack((np.arange(ex.shape[0]), true_s))
    true_dist = np.zeros(ex.shape)
    # ind = np.vstack((np.arange(ex.shape[0]), true_s))
    true_dist[np.arange(ex.shape[0]), true_s.astype(int)] = 1 # true_s + 1???

    CE = scipy.stats.entropy(true_dist, base=base, axis=1) + scipy.stats.entropy(true_dist, ex, base=base, axis=1)

    if time_average:
        return np.average(CE)

    if time_sum:
        return np.sum(CE)

    return CE


if __name__ == "__main__":
    trials = 500
    trials_to_plot = 5
    T = 100
    r=10

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8),
                                   gridspec_kw={'height_ratios': [4, 1]})

    CE_sum = np.zeros(T)

    colors = plt.cm.coolwarm(np.linspace(1, 0, trials_to_plot))

    for trial in range(trials):
        ex, bpred_s, states = step_HMM_inference({
            'r': r,
            'T': T,
            'sigma': 0.35
        }, compress_states=False)

        bex = compress_states(ex)
        bstates = (states == r).astype(int)

        CE_sum += cross_entropy(bex, bstates) # use binary cross entropy as distinguishing between individual MC's "does not matter"

        # Main plot (top)
        # ax1.matshow(ex.T)

        if trial < trials_to_plot:
            alpha = 1 if trial == 0 else 0.25
            color = colors[trial]
            color = 'black' if trial == 0 else color

            #ax1.plot(np.arange(T), bpred_s, color=color, linestyle="dashed",
            #         label=f'Trial {trial + 1} (Predicted)', alpha=alpha)
            ax1.plot(np.arange(T), states, color=color, label=f'Trial {trial + 1} (True)', alpha=alpha)
            # plot prediction marker

            first_pred_index = np.argmax(bpred_s == 1) if np.any(bpred_s==1) else None

            if first_pred_index is not None:
                ax1.scatter(first_pred_index, states[first_pred_index],
                            color=color, marker='d', s=100, alpha=alpha,
                            label=f'Predicted Jump Time (Trial {trial + 1})',
                            zorder=3)

            '''if trial == 1: # plot mode
                modes = np.argmax(ex, axis=1) # + 1
                ax1.plot(np.arange(T), modes, color=color, label=f'Trial {trial + 1} (Mode)', alpha=alpha, linestyle='dotted')
'''
        print('completed', trial)

    # Cross-entropy plot (bottom)
    ax2.plot(np.arange(T), CE_sum / trials, color='black', linewidth=2)
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Cross-Entropy')
    ax2.set_title('Average Cross-Entropy Over Time (' + str(trials) + ' trials)')

    # Add legend to the top plot (reduce redundancy with a single entry per trial)
    handles, labels = ax1.get_legend_handles_labels()
    unique_labels = dict(zip(labels, handles))  # Remove duplicates
    ax1.set_title("Markov state (s) over time")
    ax1.set_ylabel("Markov state (proportional to rate)")
    ax1.legend(unique_labels.values(), unique_labels.keys(),
               loc='best', ncol=2, framealpha=0.5)

    plt.tight_layout()
    plt.savefig('./plots/step_inference_traces_mode.png')
