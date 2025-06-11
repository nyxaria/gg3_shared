import numpy as np
import models
import scipy
import inference
from models_hmm import RampModelHMM
import os
import matplotlib.pyplot as plt

import task_2_3


num_trials = 600
num_samples = 15 # on the grid
K = 50
save_filename = "./results/CE_15x0x15rh.npy" # TODO

    # TODO
T = 100
ms = np.linspace(T * 0.1, T * 0.75, num_samples)
rs = np.rint(np.linspace(0, 20, num_samples+1))[1:]
betas = np.linspace(0, 4, num_samples)
sigmas = np.linspace(0, 4, num_samples)
x0s = np.linspace(0.2, 1, num_samples)
Rhs = np.linspace(0, 500, num_samples)

# TODO changeme
ivar1 = x0s # x-axis
ivar2 = Rhs

print("loading " + save_filename)
CE_mat = np.load(save_filename)



iv1_grid, iv2_grid = np.meshgrid(ivar1, ivar2, indexing='ij')
iv1_flat = iv1_grid.flatten()
iv2_flat = iv2_grid.flatten()
CE_flat = CE_mat.flatten()

os.makedirs('plots', exist_ok=True)


plt.figure(figsize=(8 * 0.6, 6 * 0.6))
contour = plt.tricontourf(iv1_flat, iv2_flat, CE_flat, levels=20, cmap='viridis')
plt.colorbar(contour, label='Cross-Entropy (CE)')
plt.xlabel('x0') # TODO
plt.ylabel('Rh') # TODO
plt.title(r'Ramp: CE, varying $x_0, Rh$') # TODO
plt.savefig('plots/ramp_x0Rh_smol.png')
plt.show()
