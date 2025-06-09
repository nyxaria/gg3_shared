import numpy as np
import matplotlib.pyplot as plt
import models_hmm
import numpy.random as npr
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

T=1000
m = T/2
x0 = 0.2
Rh = 50
bin_edges=np.linspace(0,1000,20)
num_trajectories=5
colors = ['blue', 'orange', 'green', 'red']

# plot trajectories and jump time histograms for simple 2-state model
# jumps histograms for simulate_2state should look like the 
# Week 1 histograms that have r=1 since Geom(p) = NB(1,p).
fig1 = plt.figure()
fig2 = plt.figure()
rs = np.linspace(1, 10, 4, dtype=int)
i=0
for r in rs:
    step = models_hmm.StepModelHMM(m, r, x0, Rh)
    spikes, jumps, rates = step.simulate_2state(Ntrials=100, T=T)
    color = colors[i]
    plt.figure(fig1.number)
    for j in range(num_trajectories):
        if j==0:
            label = 'r='+str(r)
        else:
            label=None
        plt.plot(rates[j], color=color, label=label)
    plt.figure(fig2.number)
    plt.hist(jumps, bins=bin_edges, alpha=0.5, label='r='+str(r))
    i+=1

plt.figure(fig1.number)
plt.xlabel('Time (ms)')
plt.ylabel('Firing Rate (Hz)')
plt.title('m=' + str(np.round(m)))
plt.legend()

plt.figure(fig2.number)
plt.xlabel('Jump Time (ms)')
plt.ylabel('Frequency')
plt.title('m=' + str(np.round(m)))
plt.legend()

plt.show()

# plot trajectories and jump time histograms for exact model with r+1 states
fig1 = plt.figure()
fig2 = plt.figure()
rs = np.linspace(1,100,4, dtype=int)
i = 0
for r in rs:
    step = models_hmm.StepModelHMM(m, r, x0, Rh)
    spikes, jumps, rates = step.simulate_exact(Ntrials=100, T=T)
    color = colors[i]
    plt.figure(fig1.number)
    for j in range(num_trajectories):
        if j==0:
            label = 'r='+str(r)
        else:
            label=None
        plt.plot(rates[j], color=color, label=label)
    plt.figure(fig2.number)
    plt.hist(jumps, bins=bin_edges, alpha=0.5, label='r='+str(r))
    i+=1

plt.figure(fig1.number)
plt.xlabel('Time (ms)')
plt.ylabel('Firing Rate (Hz)')
plt.title('m=' + str(np.round(m)))
plt.legend()

plt.figure(fig2.number)
plt.xlabel('Jump Time (ms)')
plt.ylabel('Frequency')
plt.title('m=' + str(np.round(m)))
plt.legend()

plt.show()

# plot trajectories and jump time histograms for exact 2-state model (inhomogeneous MC)
fig1 = plt.figure()
fig2 = plt.figure()
rs = np.linspace(1, 100, 4, dtype=int)
i = 0
for r in rs:
    step = models_hmm.StepModelHMM(m, r, x0, Rh)
    spikes, jumps, rates = step.simulate_exact_2state(Ntrials=100, T=T)
    color = colors[i]
    plt.figure(fig1.number)
    for j in range(num_trajectories):
        if j==0:
            label = 'r='+str(r)
        else:
            label=None
        plt.plot(rates[j], color=color, label=label)
    plt.figure(fig2.number)
    plt.hist(jumps, bins=bin_edges, alpha=0.5, label='r='+str(r))
    i+=1

plt.figure(fig1.number)
plt.xlabel('Time (ms)')
plt.ylabel('Firing Rate (Hz)')
plt.title('m=' + str(np.round(m)))
plt.legend()

plt.figure(fig2.number)
plt.xlabel('Jump Time (ms)')
plt.ylabel('Frequency')
plt.title('m=' + str(np.round(m)))
plt.legend()

plt.show()