import numpy as np
import models
import scipy
import inference
from models_hmm import RampModelHMM
import matplotlib.pyplot as plt

K = 50  # Markov size
Rh = 100
x0 = 0.2
beta = 0.5
T = 100
filter = True


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


def cross_entropy(ex, true_s, base=2, time_average=False):
    assert ex.shape[0] == len(true_s)

    # delta_indices = np.column_stack((np.arange(ex.shape[0]), true_s))
    true_dist = np.zeros(ex.shape)
    true_dist[np.arange(ex.shape[0]), true_s] = 1 # true_s + 1???

    CE = scipy.stats.entropy(true_dist, base=base, axis=1) + scipy.stats.entropy(true_dist, ex, base=base, axis=1)

    if time_average:
        return np.average(CE)

    return CE


if __name__ == "__main__":
    trials = 50
    trials_to_plot = 5
    T = 100
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8),
                                   gridspec_kw={'height_ratios': [4, 1]})

    CE_sum = np.zeros(T)

    colors = plt.cm.coolwarm(np.linspace(1, 0, trials_to_plot))

    for trial in range(trials):
        ex, expected_s, states = ramp_HMM_inference({
            'T': T,
            'sigma': 0.35
        })

        # Create a figure with two subplots (2 rows, 1 column)

        # true_s = (xs * (K - 1)).flatten().astype(int)
        CE_sum += cross_entropy(ex, states)

        # Main plot (top)
        # ax1.matshow(ex.T)

        if trial < trials_to_plot:
            alpha = 1 if trial == 1 else 0.25
            color = colors[trial]
            color = 'black' if trial == 1 else color
            ax1.plot(np.arange(T), expected_s, color=color, linestyle="dashed",
                     label=f'Trial {trial + 1} (Predicted)', alpha=alpha)
            ax1.plot(np.arange(T), states, color=color, label=f'Trial {trial + 1} (True)', alpha=alpha)

            if trial == 1: # plot mode
                modes = np.argmax(ex, axis=1) # + 1
                ax1.plot(np.arange(T), modes, color=color, label=f'Trial {trial + 1} (Mode)', alpha=alpha, linestyle='dotted')

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
               loc='upper left', ncol=2, framealpha=0.5)

    plt.tight_layout()
    plt.savefig('./plots/ramp_inference_traces_mode.png')
