import learn_md
import torch

from pathlib import Path
from md_models import MD_ReLU, MD_Abs, MD_Sigmoid

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

seed = 47
learn_md.set_random_seeds(seed)

# Dimensionality and number of samples
N = 10
num_samples = 1000
trials=20
epochs=50000

results_dir = Path("results/md_10d/")
models_dir = results_dir / "models"
models_dir.mkdir(parents=True, exist_ok=True)

# create the parameterized Gaussian
mean, cov, eigvals_sorted, eigvecs_sorted = learn_md.create_gaussian(N,device=device)
torch.save((N, mean, cov, eigvals_sorted, eigvecs_sorted), results_dir / "data_model.pt")

# create the training data
x = torch.distributions.MultivariateNormal(mean, covariance_matrix=cov).sample((num_samples,))
y = learn_md.mahalanobis_distance(x, mean, eigvals_sorted, eigvecs_sorted)

for trial in range(trials):
    print()
    print(f"Training trial {trial}")

    # create shared starting weights and biases for the models
    W, b = learn_md.create_weights_and_biases(N, 2*N, x, device=device)

    relu_model, relu_losses, relu_stats = \
        learn_md.train_model(x, y, MD_ReLU(N, W, b), epochs=epochs, betas=(0.9, 0.99), device=device)
    abs_model, abs_losses, abs_stats = \
        learn_md.train_model(x, y, MD_Abs(N, W[:N], b[:N]), epochs=epochs, betas=(0.9, 0.99), device=device)
    sigmoid_model, sigmoid_losses, sigmoid_stats = \
        learn_md.train_model(x, y, MD_Sigmoid(N, W, b), epochs=epochs, betas=(0.9, 0.99), device=device)

    trial_dir = models_dir / str(trial)
    trial_dir.mkdir(parents=True, exist_ok=True)

    torch.save(relu_model.state_dict(), trial_dir / "relu_model.pt")
    torch.save(abs_model.state_dict(), trial_dir / "abs_model.pt")
    torch.save(sigmoid_model.state_dict(), trial_dir / "sigmoid_model.pt")

    torch.save((relu_losses, relu_stats), trial_dir / "relu_stats.pt")
    torch.save((abs_losses, abs_stats), trial_dir / "abs_stats.pt")
    torch.save((sigmoid_losses, sigmoid_stats), trial_dir / "sigmoid_stats.pt")
