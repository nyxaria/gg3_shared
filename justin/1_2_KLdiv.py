import numpy as np
import matplotlib.pyplot as plt
import math
import scipy

import torch
import torch.nn as nn
import torch.optim as optim
import sklearn

plt.style.use('ggplot')


n_psths = 10

ltrials_list = np.linspace(2, 5, 7)
trials_list = np.pow(10, ltrials_list)

divergences = []

for Ntrials in trials_list:
    psths = []

    for _ in range(n_psths):
        spikes, jumps, rates = step.simulate(Ntrials=int(Ntrials), T=100)

        counts, edges = PSTH(spikes, mode='valid')
        psths.append(counts)

    # Calculate pairwise symmetric KL divergences
    kl_matrix = np.zeros((n_psths, n_psths))
    for i in range(n_psths):
        for j in range(i+1, n_psths):
            # print(psths[i].shape, psths[j].shape)
            kl_matrix[i,j] = symmetric_kl(psths[i], psths[j])
            kl_matrix[j,i] = kl_matrix[i,j]  # Symmetric

# Average the upper triangle (excluding diagonal)
    average_kl = kl_matrix[np.triu_indices(n_psths, k=1)].mean()

    divergences += [average_kl]

print(trials_list)
print(divergences)


plt.ylabel('Log Symmetric KL Divergence')
plt.xlabel('log(trials)')
plt.title('Symmetric KL Divergence in PSTHs vs. number of trials')
plt.plot(ltrials_list, np.log(divergences))

plt.show()