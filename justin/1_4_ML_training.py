# 1.4
import numpy as np
import matplotlib.pyplot as plt
import math
import scipy

import torch
import torch.nn as nn
import torch.optim as optim
import sklearn

plt.style.use('ggplot')


rng = 69420

'''
ML model

batch summary traces dynamically to form mini-datasets (sparser datasets need more batching)

gaussian kernel smoothing (and possible averaging of n traces) of each batch

Compute summary statics for each batch

summary statistics
- PSTH
- Fano Factor
- Inter-spike interval (ISI) distribution

use a simple NN to classify the summary statistics

'''

# generate data

T = 100
num_datapoints = 500000
# Ntrials = 100000 // 2

grid_samples = 10 # take 10 samples along the range of each parameter

# parameter ranges for the train data
beta_range = [0, 4]
log_sigma_range = np.log([0.04, 4])
r_range = [0.5, 6]
m_range = [T * 0.25, T * 0.75]
x0_range = [0, 0.5]

# generate step data using a grid along the parameter space {m, r, x0}
assert grid_samples ** 3 < 0.5 * num_datapoints
samples_per_param_set = (0.5 * num_datapoints) // (grid_samples ** 3)
samples_per_param_set = int(samples_per_param_set)

print('generating', (samples_per_param_set * 2 * grid_samples ** 3), 'samples out of the desired', num_datapoints)

# step

r_values = np.linspace(*r_range, grid_samples)
m_values = np.linspace(*m_range, grid_samples)
x0_values = np.linspace(*x0_range, grid_samples)

# permute
M, R, X0 = np.meshgrid(m_values, r_values , x0_values, indexing='ij')
combinations = np.stack((M, R, X0), axis=-1)
combinations = np.reshape(combinations, (grid_samples ** 3, 3))

step_spikes = []

for combination in combinations:
    combination_step_spikes, _ = models.StepModel(*combination).simulate(Ntrials=samples_per_param_set, T=T, get_rate=False)
    step_spikes.append(combination_step_spikes)

step_spikes = np.concatenate(step_spikes)

# ramp

beta_values = np.linspace(*beta_range, grid_samples)
sigma_values = np.linspace(*log_sigma_range, grid_samples)
sigma_values = np.exp(sigma_values)
x0_values = np.linspace(*x0_range, grid_samples)

# permute
B, s, X0 = np.meshgrid(beta_values, sigma_values , x0_values, indexing='ij')
combinations = np.stack((B, s, X0), axis=-1)
combinations = np.reshape(combinations, (grid_samples ** 3, 3))

ramp_spikes = []

for combination in combinations:
    combination_ramp_spikes, _ = models.RampModel(*combination).simulate(Ntrials=samples_per_param_set, T=T, get_rate=False)
    ramp_spikes.append(combination_ramp_spikes)

ramp_spikes = np.concatenate(ramp_spikes)

print(step_spikes.shape, ramp_spikes.shape)


# step_spikes, _ = models.StepModel().simulate(Ntrials=Ntrials, T=T, get_rate=False)
# ramp_spikes, _ = models.RampModel().simulate(Ntrials=Ntrials, T=T, get_rate=False)

# TODO: dynamically determine batch size
batch_size = 20
num_batches = (samples_per_param_set * grid_samples ** 3) // batch_size

X_step = compute_summary_statistics(step_spikes, batch_size)
X_ramp = compute_summary_statistics(ramp_spikes, batch_size)

X = np.concatenate((X_step, X_ramp))
y = np.concatenate((np.zeros(num_batches), np.ones(num_batches)))
X, y = sklearn.utils.shuffle(X , y, random_state=rng)

print(X.shape, y.shape)

X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.2, random_state=rng)

scaler = sklearn.preprocessing.StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)
y_train = torch.FloatTensor(y_train)
y_test = torch.FloatTensor(y_test)


class StepRampClassifier(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(input_dim, 15),
            nn.ReLU(),
            nn.Linear(15, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.layers(x)


model = StepRampClassifier(input_dim=X_train.shape[1])

# Define loss function and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 50
ml_batch_size = 10

# Train the model

# Initialize lists to store metrics
train_losses = []
test_accuracies = []

# Training loop
for epoch in range(epochs):
    model.train()  # Set model to training mode
    epoch_loss = 0.0

    # Mini-batch training
    for i in range(0, len(X_train), ml_batch_size):
        # Get mini-batch
        inputs = X_train[i:i+ml_batch_size]
        labels = y_train[i:i+ml_batch_size]

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)

        loss = criterion(outputs, labels.unsqueeze(1))
        epoch_loss += loss.item()

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

    # Calculate average training loss for the epoch
    avg_epoch_loss = epoch_loss / (len(X_train) / ml_batch_size)
    train_losses.append(avg_epoch_loss)

    # Evaluate on test set
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():
        test_outputs = model(X_test)
        predicted = test_outputs.round()
        correct = (predicted == y_test.unsqueeze(1)).sum().item()
        accuracy = correct / len(y_test) * 100
        test_accuracies.append(accuracy)

    # Print progress
    if (epoch+1) % 20 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_epoch_loss:.4f}, Test Accuracy: {accuracy:.2f}%')

# Plotting the results
plt.figure(figsize=(12, 5))

# Plot training loss
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss over Epochs')
plt.legend()

# Plot test accuracy
plt.subplot(1, 2, 2)
plt.plot(test_accuracies, label='Test Accuracy', color='orange')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Test Accuracy over Epochs')
plt.legend()

plt.tight_layout()
plt.show()



# Save to a specific folder in Drive
save_path = './results/StepRamp.pth'
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scaler_state': scaler
}, save_path)