import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from bayesianNet import BayesianNet
import matplotlib.pyplot as plt


np.random.seed(42)


# Create an instance of the Bayesian neural network
model = BayesianNet()

# Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Generate an arbitrary dataset
n_train = 50
std_train = 0.5
X = np.sort(np.random.uniform(-np.pi, np.pi, n_train)).astype(np.float32).reshape(n_train,1)
y = np.sin(X) + np.random.normal(0, std_train, n_train).astype(np.float32)

# Convert the dataset to PyTorch tensors
X_tensor = torch.from_numpy(X)
y_tensor = torch.from_numpy(y)

# Train the model for 1000 epochs
for epoch in range(10000):
    # Forward pass
    output = model(X_tensor)
    
    
    # Compute the loss
    loss = criterion(output, y_tensor)

    # Compute the KL divergence loss for the Bayesian layers
    # kl_divergence_loss = model.layer1.compute_kl_divergence_loss() + model.layer2.compute_kl_divergence_loss()

    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    # (loss + kl_divergence_loss).backward()
    optimizer.step()

    # Ensure that the bias vectors are sorted in descending order
    model.layer1.sort_bias()
    model.layer2.sort_bias()

    # Print the loss every 100 epochs
    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch + 1}: Loss = {loss.item():.4f}")


# Generate test data


n_test = 1000
std_test = 0.1
x_test = np.sort(np.random.uniform(-np.pi, np.pi, n_test))
y_test = np.sin(x_test) + np.random.normal(0, std_test, n_test)

n_runs = 1000

# Evaluate function using the Bayesian network
    
y_preds = np.zeros((n_runs, np.size(x_test)))
for i in range(n_runs):
    y_pred = model.forward(torch.Tensor(x_test).unsqueeze(1)).detach().numpy().flatten()
    y_preds[i] = y_pred

# Calculate mean and quantiles
mean_y = np.mean(y_preds, axis=0)
lower_y = np.quantile(y_preds, 0.05, axis=0)
upper_y = np.quantile(y_preds, 0.95, axis=0)

# Plot results
plt.plot(x_test, y_test, label='True Function')
plt.plot(x_test, mean_y, label='Average Prediction')
plt.fill_between(x_test, lower_y, upper_y, alpha=0.5, label='5%-95% Quantile')
plt.legend()
plt.show()