import torch
import matplotlib.pyplot as plt
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import time
import os

from models_md import MD_ReLU, MD_Abs, MD_Sigmoid

def create_gaussian(dim, device):
    # Create a multivariate normal distribution with random mean in [-10, 10]
    mean = (torch.rand(dim, device=device) * 20) - 10

    A = torch.randn(dim, dim, device=device)
    cov = A @ A.T  # Make it symmetric and positive semi-definite
    cov += 1e-3 * torch.eye(dim, device=device)

    eigvals, eigvecs = torch.linalg.eigh(cov)  # Returns in ascending order by default

    # Sort in descending order (largest eigenvalue first)
    sorted_indices = torch.argsort(eigvals, descending=True)
    eigvals_sorted = eigvals[sorted_indices]
    eigvecs_sorted = eigvecs[:, sorted_indices]

    # Scale the covariarnce matrix so that the largest eigenvalue is 1.0
    max_eigval = eigvals_sorted[0]
    cov = cov / max_eigval
    eigvals_sorted = eigvals_sorted / max_eigval

    print(f"eigvals_sorted = {eigvals_sorted}")
    print(f"eigvecs_sorted = {eigvecs_sorted}")

    is_symmetric = torch.allclose(cov, cov.T, atol=1e-6)
    is_psd = torch.all(eigvals_sorted >= 0)
    print(f"is_symmetric = {is_symmetric}, is_psd  = {is_psd}")

    return mean, cov, eigvals_sorted, eigvecs_sorted

def mahalanobis_distance(x, mean, eigvals_sorted, eigvecs_sorted):
    centered = x - mean  # shape: (num_samples, N)
    proj = centered @ eigvecs_sorted  # shape: (num_samples, N)
    scaled = proj / torch.sqrt(eigvals_sorted)  # element-wise division, shape: (num_samples, N)
    md = torch.norm(scaled, dim=1)  # Mahalanobis distance for each sample
    return md

def plot_eigenvalues(eigvals_sorted):
    # Plot eigenvalues
    plt.figure(figsize=(8, 4))
    plt.plot(eigvals_sorted.cpu().numpy(), marker='o')
    plt.title("Eigenvalues (variance)")
    plt.xlabel("Component Index")
    plt.ylabel("Eigenvalues")
    plt.xticks(range(len(eigvals_sorted)))  # Only show integer x-ticks
    plt.ylim(0.0, 1.1)  # Fixed y-axis range
    plt.grid(True, axis='y')
    plt.tight_layout()

def top_projection(data, mean, eigvecs_sorted):
    V = eigvecs_sorted[:, :2]  # Shape: (N, 2)
    print(f"V = {V}")
    proj_2d = ((data - mean) @ V)  # shape (num_samples, 2)
    data_in_pc_plane = proj_2d @ V.T + mean
    return data_in_pc_plane

def train_model(x, y, model, betas, epochs=10000, lr=1e-3, device=None, verbose=True):
    print(f"Training started for {type(model).__name__} with beta={betas}")

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    x = x.to(device)
    y = y.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=betas)
    loss_fn = nn.MSELoss()

    losses = []
    start_time = time.time()

    for epoch in range(epochs):
        model.train()

        # Forward pass
        y_pred = model(x)

        # Compute loss
        loss = loss_fn(y_pred, y)
        losses.append(loss.item())

        # Backward + optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if verbose and (epoch % 500 == 0 or epoch == epochs - 1):
            print(f"Epoch {epoch:4d} | Loss: {loss.item():.6f}")

    elapsed = time.time() - start_time

    if verbose:
        print(f"Training completed in {elapsed:.2f} seconds.")
        print()

    return model, losses, {
        "final_loss": losses[-1],
        "total_time_sec": elapsed,
        "device": str(device),
        "epochs": epochs,
    }

def create_weights_and_biases(input_dim, base_count, data, device):
    W = torch.empty(base_count, input_dim, device=device)
    torch.nn.init.kaiming_normal_(W, nonlinearity='relu')

    # Sample base_count points from data
    indices = torch.randint(0, data.shape[0], (base_count,), device=device)
    X = data[indices]  # shape: (base_count, input_dim)

    # Compute b = -W @ x_i for each i
    b = -torch.sum(W * X, dim=1)  # element-wise dot product row-wise

    return W, b

def set_random_seeds(seed):
    # Set random seeds for reproducibility
    seed = 47
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    print(f"Seed = {seed}")

def plot_losses(*loss_data, labels=None):
    plt.figure(figsize=(8, 5))
    for i, losses in enumerate(loss_data):
        label = labels[i] if labels else f"Model {i+1}"
        plt.plot(losses, label=label)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.yscale("log")
    plt.title("Training Loss over Epochs")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

# # Plot eigenvalues
# plot_eigenvalues(eigvals_sorted)

# # Plot top two components
# data_in_pc_plane = top_projection(x, mean, eigvecs_sorted)
# plot_2d_data(data_in_pc_plane)

# plt.show()

# plot_losses(relu_losses, abs_losses, sigmoid_losses, labels=["ReLU", "Abs", "Sigmoid"])
