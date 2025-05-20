# xor_train.py

import torch
import xor_lib
from xor_models import XOR_ReLU, XOR_Abs, XOR_Sigmoid
from pathlib import Path

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set random seed for reproducibility
seed = 47
xor_lib.set_random_seeds(seed)

# Output directories
results_dir = Path("results/xor/")
models_dir = results_dir / "models"
models_dir.mkdir(parents=True, exist_ok=True)

# Create XOR dataset
x, y = xor_lib.create_xor_dataset()
x = x.to(device)
y = y.to(device)

# Experiment settings
trials = 20
epochs = 5000  # Enough for convergence on 4 points

for trial in range(trials):
    print()
    print(f"Training trial {trial}")

    # Create models
    relu_model = XOR_ReLU()
    abs_model = XOR_Abs()
    sigmoid_model = XOR_Sigmoid()

    # Train each model
    print("ReLU:")
    relu_model, relu_losses, relu_stats = xor_lib.train_model(relu_model, x, y, epochs=epochs, device=device)
    print("Abs:")
    abs_model, abs_losses, abs_stats = xor_lib.train_model(abs_model, x, y, epochs=epochs, device=device)
    print("Sigmoid:")
    sigmoid_model, sigmoid_losses, sigmoid_stats = xor_lib.train_model(sigmoid_model, x, y, epochs=epochs, device=device)

    # Create trial folder
    trial_dir = models_dir / str(trial)
    trial_dir.mkdir(parents=True, exist_ok=True)

    # Save model states
    torch.save(relu_model.state_dict(), trial_dir / "relu_model.pt")
    torch.save(abs_model.state_dict(), trial_dir / "abs_model.pt")
    torch.save(sigmoid_model.state_dict(), trial_dir / "sigmoid_model.pt")

    # Save training losses and stats
    torch.save((relu_losses, relu_stats), trial_dir / "relu_stats.pt")
    torch.save((abs_losses, abs_stats), trial_dir / "abs_stats.pt")
    torch.save((sigmoid_losses, sigmoid_stats), trial_dir / "sigmoid_stats.pt")
