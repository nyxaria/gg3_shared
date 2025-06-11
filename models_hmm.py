import numpy as np
import numpy.random as npr

from scipy.stats import norm
from scipy.stats import nbinom

def lo_histogram(x, bins):
    """
    Left-open version of np.histogram with left-open bins covering the interval (left_edge, right_edge]
    (np.histogram does the opposite and treats bins as right-open.)
    Input & output behaviour is exactly the same as np.histogram
    """
    out = np.histogram(-x, -bins[::-1])
    return out[0][::-1], out[1:]


def gamma_isi_point_process(rate, shape):
    """
    Simulates (1 trial of) a sub-poisson point process (with underdispersed inter-spike intervals relative to Poisson)
    :param rate: time-series giving the mean spike count (firing rate * dt) in different time bins (= time steps)
    :param shape: shape parameter of the gamma distribution of ISI's
    :return: vector of spike counts with same shape as "rate".
    """
    sum_r_t = np.hstack((0, np.cumsum(rate)))
    gs = np.zeros(2)
    while gs[-1] < sum_r_t[-1]:
        gs = np.cumsum( npr.gamma(shape, 1 / shape, size=(2 + int(2 * sum_r_t[-1]),)) )
    y, _ = lo_histogram(gs, sum_r_t)

    return y


class StepModelHMM():
    """
    Simulator of the Stepping Model of Latimer et al. Science 2015.
    """
    def __init__(self, m=50, r=10, x0=0.2, Rh=50, isi_gamma_shape=None, Rl=None, dt=None):
        """
        Simulator of the Stepping Model of Latimer et al. Science 2015.
        :param m: mean jump time (in # of time-steps). This is the mean parameter of the Negative Binomial distribution
                  of jump (stepping) time
        :param r: parameter r ("# of successes") of the Negative Binomial (NB) distribution of jump (stepping) time
                  (Note that it is more customary to parametrise the NB distribution by its parameter p and r,
                  instead of m and r, where p is so-called "probability of success" (see Wikipedia). The two
                  parametrisations are equivalent and one can go back-and-forth via: m = r (1-p)/p and p = r / (m + r).)
        :param x0: determines the pre-jump firing rate, via  R_pre = x0 * Rh (see below for Rh)
        :param Rh: firing rate of the "up" state (the same as the post-jump state in most of the project tasks)
        :param isi_gamma_shape: shape parameter of the Gamma distribution of inter-spike intervals.
                            see https://en.wikipedia.org/wiki/Gamma_distribution
        :param Rl: firing rate of the post-jump "down" state (rarely used)
        :param dt: real time duration of time steps in seconds (only used for converting rates to units of inverse time-step)
        """
        self.m = m
        self.r = r
        self.x0 = x0

        self.p = r / (m + r)

        self.Rh = Rh
        if Rl is not None:
            self.Rl = Rl

        self.isi_gamma_shape = isi_gamma_shape
        self.dt = dt


    @property
    def params(self):
        return self.m, self.r, self.x0

    @property
    def fixed_params(self):
        return self.Rh, self.Rl


    def emit(self, rate):
        """
        emit spikes based on rates
        :param rate: firing rate sequence, r_t, possibly in many trials. Shape: (Ntrials, T)
        :return: spike train, n_t, as an array of shape (Ntrials, T) containing integer spike counts in different
                 trials and time bins.
        """
        if self.isi_gamma_shape is None:
            # poisson spike emissions
            y = npr.poisson(rate * self.dt)
        else:
            # sub-poisson/underdispersed spike emissions
            y = gamma_isi_point_process(rate * self.dt, self.isi_gamma_shape)

        return y

    # TODO: maybe separate the transition matrix code, if 2.3/further sections need it
    def simulate_2state(self, Ntrials=1, T=100, get_rate=True):
        """
        :param Ntrials: (int) number of trials
        :param T: (int) duration of each trial in number of time-steps.
        :param get_rate: whether or not to return the rate time-series
        :return:
        spikes: shape = (Ntrial, T); spikes[j] gives the spike train, n_t, in trial j, as
                an array of spike counts in each time-bin (= time step)
        jumps:  shape = (Ntrials,) ; jumps[j] is the jump time (aka step time), tau, in trial j.
        rates:  shape = (Ntrial, T); rates[j] is the rate time-series, r_t, in trial j (returned only if get_rate=True)
        """
        # set dt (time-step duration in seconds) such that trial duration is always 1 second, regardless of T.
        dt = 1 / T
        self.dt = dt

        # in this model, the jump times, tau, follow Geom(p) rather than NB(r,p) so it is not exact
        transition = np.array([[1-self.p, self.p], [0, 1]])

        spikes, jumps, rates = [], [], []
        for tr in range(Ntrials):
            state = 0 # start with initial state = x0
            rate = np.ones(T)*self.Rh
            rate[0] = rate[0]*self.x0
            for t in range(T-1):
                sample = npr.binomial(1,transition[state][state+1])
                state+=sample
                if state==1:
                    break
                else:
                    rate[t+1]*=self.x0

            jumps.append(np.argmax(rate))
            rates.append(rate)
            spikes.append(self.emit(rate))

        if get_rate:
            return np.array(spikes), np.array(jumps), np.array(rates)
        else:
            return np.array(spikes), np.array(jumps)

    def _calculate_transition_matrix_exact(self, Ntrials=1, T=100):
        transition = np.identity(int(self.r) + 1)
        for i in range(int(self.r)):
            transition[i][i] = 1 - self.p
            transition[i][i + 1] = self.p
        return transition

    def _calculate_initial_distribution_exact(self):
        # always start from the first state
        pi = np.zeros(int(self.r) + 1)
        pi[0] = 1
        return pi

    def simulate_exact(self, Ntrials=1, T=100, get_rate=True, return_state_indices=False, delay_compensation=False):

        """
        :param Ntrials: (int) number of trials
        :param T: (int) duration of each trial in number of time-steps.
        :param get_rate: whether or not to return the rate time-series
        :return:
        spikes: shape = (Ntrial, T); spikes[j] gives the spike train, n_t, in trial j, as
                an array of spike counts in each time-bin (= time step)
        jumps:  shape = (Ntrials,) ; jumps[j] is the jump time (aka step time), tau, in trial j.
        rates:  shape = (Ntrial, T); rates[j] is the rate time-series, r_t, in trial j (returned only if get_rate=True)
        """
        # set dt (time-step duration in seconds) such that trial duration is always 1 second, regardless of T.
        dt = 1 / T
        self.dt = dt

        # in this model, the states are 0 <= number of successes <= r
        # in this model, jump occurs after rth success so it is delayed from Week 1 model by r time-steps
        transition = self._calculate_transition_matrix_exact(Ntrials, T)

        spikes, jumps, rates, states = [], [], [], []
        for tr in range(Ntrials):



            sim_steps = T+self.r-1 if delay_compensation else T-1

            state = np.ones(sim_steps+1) * self.r # states vector for current trial. Assume all are at the max state
            rate = np.ones(sim_steps+1)*self.Rh

            rate[0] = rate[0]*self.x0
            cur_state = 0  # start with initial state = x0
            state[0] = cur_state


            for t in range(sim_steps):

                sample = npr.binomial(1,transition[cur_state][cur_state+1])
                cur_state+=sample
                if cur_state==self.r:
                    break
                else:
                    rate[t+1]*=self.x0

                state[t+1] = cur_state

            if delay_compensation:
                rate = rate[self.r:]
                state = state[self.r:]

            jumps.append(np.argmax(rate))
            rates.append(rate)
            spikes.append(self.emit(rate))
            states.append(state)

        if return_state_indices:
            rates = states

        if get_rate:
            return np.array(spikes), np.array(jumps), np.array(rates)
        else:
            return np.array(spikes), np.array(jumps)

    def simulate_exact_2state(self, Ntrials=1, T=100, get_rate=True):
        """
        :param Ntrials: (int) number of trials
        :param T: (int) duration of each trial in number of time-steps.
        :param get_rate: whether or not to return the rate time-series
        :return:
        spikes: shape = (Ntrial, T); spikes[j] gives the spike train, n_t, in trial j, as
                an array of spike counts in each time-bin (= time step)
        jumps:  shape = (Ntrials,) ; jumps[j] is the jump time (aka step time), tau, in trial j.
        rates:  shape = (Ntrial, T); rates[j] is the rate time-series, r_t, in trial j (returned only if get_rate=True)
        """
        # set dt (time-step duration in seconds) such that trial duration is always 1 second, regardless of T.
        dt = 1 / T
        self.dt = dt

        spikes, jumps, rates = [], [], []
        for tr in range(Ntrials):
            state = 0 # start with initial state = x0
            rate = np.ones(T)*self.Rh
            rate[0] = rate[0]*self.x0
            for t in range(T-1):
                # in this model, p_t = P(rth success occurs after exactly t failures|rth success occurs after > t-1 failures)
                p_t = nbinom.pmf(t, self.r, self.p)/(1-nbinom.cdf(t, self.r, self.p))
                transition = np.array([[1-p_t, p_t], [0, 1]])
                sample = npr.binomial(1,np.clip(transition[state][state+1], 0, 1))
                state+=sample
                if state==1:
                    break
                else:
                    rate[t+1]*=self.x0

            jumps.append(np.argmax(rate))
            rates.append(rate)
            spikes.append(self.emit(rate))

        if get_rate:
            return np.array(spikes), np.array(jumps), np.array(rates)
        else:
            return np.array(spikes), np.array(jumps)

class RampModelHMM:
    """
    Simulator of the HMM approximation of the Ramping Model of Latimer et al., Science (2015).
    """
    def __init__(self, beta=0.5, sigma=0.2, x0=0.2, K=50, Rh=50, isi_gamma_shape=None, Rl=None, dt=None):
        """
        Simulator of the HMM approximation of the Ramping Model of Latimer et al., Science (2015).
        :param beta: drift rate of the drift-diffusion process
        :param sigma: diffusion strength of the drift-diffusion process.
        :param x0: average initial value of latent variable x[0]
        :param K: number of discrete states for x_t
        :param Rh: the maximal firing rate obtained when x_t reaches 1 (corresponding to the same as the post-step
                   state in most of the project tasks)
        :param isi_gamma_shape: shape parameter of the Gamma distribution of inter-spike intervals.
                            see https://en.wikipedia.org/wiki/Gamma_distribution
        :param Rl: Not implemented. Ignore.
        :param dt: real time duration of time steps in seconds (only used for converting rates to units of inverse time-step)
        """
        self.beta = beta
        self.sigma = sigma
        self.x0 = x0

        self.Rh = Rh
        if Rl is not None:
            self.Rl = Rl

        self.isi_gamma_shape = isi_gamma_shape
        self.dt = dt

        self.K = K

        if self.K > 0:
            self.x_values = np.linspace(0, 1, self.K)
        else:
            self.x_values = np.array([])

    @property
    def params(self):
        return self.beta, self.sigma, self.x0, self.K, self.Rh, self.Rl, self.dt

    @property
    def fixed_params(self):
        return self.Rh, self.Rl


    def f_io(self, xs, b=None):
        if b is None:
            return self.Rh * np.maximum(0, xs)
        else:
            return self.Rh * b * np.log(1 + np.exp(xs / b))
        
    def emit(self, rate):
        """
        emit spikes based on rates
        :param rate: firing rate sequence, r_t, possibly in many trials. Shape: (Ntrials, T)
        :return: spike train, n_t, as an array of shape (Ntrials, T) containing integer spike counts in different
                 trials and time bins.
        """
        if self.isi_gamma_shape is None:
            # poisson spike emissions
            y = npr.poisson(rate * self.dt)
        else:
            # sub-poisson/underdispersed spike emissions
            y = gamma_isi_point_process(rate * self.dt, self.isi_gamma_shape)

        return y

    def _get_bin_edges(self):
        bin_width_internal = 1.0 / (self.K - 1)
        edges = np.zeros(self.K + 1)
        edges[0] = -np.inf
        edges[self.K] = np.inf
        edges[1:-1] = self.x_values[:-1] + bin_width_internal / 2.0
        return edges

    def _calculate_initial_distribution(self, T=None):

        if T is not None:
            self.dt = 1.0/T

        pi_values = np.zeros(self.K)
        mu_init = self.x0
        std_init = self.sigma * np.sqrt(self.dt)
        edges = self._get_bin_edges()

        for s_idx in range(self.K):
            lower_bound = edges[s_idx]
            upper_bound = edges[s_idx+1]
            prob = 0.0
            # if sigma is too low this breaks
            if std_init > 1e-9: 
                prob = norm.cdf(upper_bound, loc=mu_init, scale=std_init) - \
                              norm.cdf(lower_bound, loc=mu_init, scale=std_init)
            else: 
                if lower_bound <= mu_init < upper_bound:
                    prob = 1.0
            pi_values[s_idx] = prob

        pi = pi_values / np.sum(pi_values)
        return pi

    def _calculate_transition_matrix(self, T=None):
        if T is not None:
            self.dt = 1.0/T

        T_matrix = np.zeros((self.K, self.K))
        std_transition = self.sigma * np.sqrt(self.dt)
        edges = self._get_bin_edges()

        for s_cur_idx in range(self.K):
            if s_cur_idx == self.K - 1: # final index is always 1
                T_matrix[s_cur_idx, self.K - 1] = 1.0
                continue

            x_curr = self.x_values[s_cur_idx]
            mu_transition = x_curr + self.beta * self.dt
            current_row_probs = np.zeros(self.K)
            
            for s_next_idx in range(self.K):
                lower_bound = edges[s_next_idx]
                upper_bound = edges[s_next_idx+1]
                prob = 0.0
                # if sigma is too low this breaks
                if std_transition > 1e-9:
                    prob = norm.cdf(upper_bound, loc=mu_transition, scale=std_transition) - \
                                  norm.cdf(lower_bound, loc=mu_transition, scale=std_transition)
                else:
                    if lower_bound <= mu_transition < upper_bound:
                         prob = 1.0
                current_row_probs[s_next_idx] = prob
            
            sum_row_probs = np.sum(current_row_probs)
            T_matrix[s_cur_idx, :] = current_row_probs / sum_row_probs
        
        return T_matrix

    def simulate(self, Ntrials=1, T=100, get_rate=True, return_state_indices=False):
        """
        :param Ntrials: (int) number of trials

        :param T: (int) duration of each trial in number of time-steps.
        :param get_rate: whether or not to return the rate time-series
        :return:
        spikes: shape = (Ntrial, T); spikes[j] gives the spike train, n_t, in trial j, as
                an array of spike counts in each time-bin (= time step)
        xs:     shape = (Ntrial, T); xs[j] is the latent variable time-series x_t in trial j
        rates:  shape = (Ntrial, T); rates[j] is the rate time-series, r_t, in trial j (returned only if get_rate=True)
        """

        # return state indices: will return state_indices in place of xs

        self.dt = 1.0 / T
        
        init_distribution = self._calculate_initial_distribution()
        transition_matrices = self._calculate_transition_matrix()

        state_indices = np.zeros((Ntrials, T), dtype=int)
        
        for trial_idx in range(Ntrials):
            init_distribution = init_distribution / np.sum(init_distribution)
            state_indices[trial_idx, 0] = np.random.choice(self.K, p=init_distribution)
            
            for t_idx in range(T - 1):
                current_s = state_indices[trial_idx, t_idx]
                next_s_probs = transition_matrices[current_s, :].copy()
                next_s_probs = next_s_probs / np.sum(next_s_probs)
                state_indices[trial_idx, t_idx+1] = np.random.choice(self.K, p=next_s_probs)
    
        xs = self.x_values[state_indices]
        rates = self.f_io(xs)

        spikes = np.zeros_like(rates, dtype=int)
        for i in range(Ntrials):
            spikes[i,:] = self.emit(rates[i,:])

        if return_state_indices:
            xs = state_indices

        if get_rate:
            return spikes, xs, rates
        else:
            return spikes, xs

