# xor_lib.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
import time

def create_xor_dataset():
    """
    Create the standard centered XOR dataset.
    
    Returns:
        x (Tensor): Shape (4,2), input points
        y (Tensor): Shape (4,), class labels {0,1}
    """
    # Inputs: (-1,-1), (1,-1), (-1,1), (1,1)
    x = torch.tensor([
        [-1.0, -1.0],
        [ 1.0, -1.0],
        [-1.0,  1.0],
        [ 1.0,  1.0]
    ], dtype=torch.float32)

    # Labels: standard XOR
    # F XOR F = F (0)
    # T XOR F = T (1)
    # F XOR T = T (1)
    # T XOR T = F (0)
    y = torch.tensor([0, 1, 1, 0], dtype=torch.long)

    return x, y

def train_model(model, x, y, epochs=5000, lr=1e-3, device=None, verbose=True):
    """
    Train a model on the XOR dataset using CrossEntropyLoss.
    
    Args:
        model (nn.Module): Model to train
        x (Tensor): Input points
        y (Tensor): Ground truth labels
        epochs (int): Number of training epochs
        lr (float): Learning rate
        device (torch.device): Device to use
        verbose (bool): Whether to print progress
    
    Returns:
        model (nn.Module): Trained model
        losses (list of float): Training loss values
        stats (dict): Final stats
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    x = x.to(device)
    y = y.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.99))
    loss_fn = nn.CrossEntropyLoss()

    losses = []
    start_time = time.time()

    for epoch in range(epochs):
        model.train()

        optimizer.zero_grad()
        outputs = model(x)  # (4,2)
        loss = loss_fn(outputs, y)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if verbose and (epoch % 500 == 0 or epoch == epochs - 1):
            print(f"Epoch {epoch:4d} | Loss: {loss.item():.6f}")

    elapsed = time.time() - start_time

    if verbose:
        print(f"Training completed in {elapsed:.2f} seconds.")

    return model, losses, {
        "final_loss": losses[-1],
        "total_time_sec": elapsed,
        "epochs": epochs,
        "device": str(device)
    }

def evaluate_model(model, x, y):
    """
    Evaluate model accuracy on XOR dataset.
    
    Args:
        model (nn.Module): Trained model
        x (Tensor): Input points
        y (Tensor): Ground truth labels
    
    Returns:
        correct (int): Number of correct predictions (0 to 4)
    """
    model.eval()
    with torch.no_grad():
        outputs = model(x)  # shape (4,2)
        preds = torch.argmax(outputs, dim=1)  # Choose highest logit
        correct = (preds == y).sum().item()

    return correct

def set_random_seeds(seed):
    """
    Set random seeds for reproducibility across torch, numpy, and random.
    
    Args:
        seed (int): Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    print(f"Random seed set to {seed}")
