import torch
import matplotlib.pyplot as plt
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import time
from scipy.optimize import linear_sum_assignment
import os
import shutil

from models_md import MD_ReLU, MD_Abs, MD_Sigmoid

def load_model(model_class, input_dim, path, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model_class(input_dim).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model

def distance_from_mean_to_hyperplanes(model, mean):
    """
    Computes the distance from the Gaussian mean to the decision boundary of each node in the model's linear layer.
    Returns a tensor of distances (one per node).
    """
    W = model.linear.weight.detach()  # shape: (out_dim, input_dim)
    b = model.linear.bias.detach()    # shape: (out_dim,)
    mean = mean.detach()              # shape: (input_dim,)

    # For each neuron: distance = |w^T * mean + b| / ||w||
    numerator = torch.abs(W @ mean + b)  # shape: (out_dim,)
    denominator = torch.norm(W, dim=1)   # shape: (out_dim,)
    distances = numerator / denominator  # shape: (out_dim,)

    return distances

def recover_sorted_eigenstructure(model):
    """
    Estimates the eigenvalues and eigenvectors from a linear model's weights,
    assuming each row of W ≈ v_i / sqrt(lambda_i).

    Returns:
        - eigvals_sorted: estimated eigenvalues (descending)
        - eigvecs_sorted: corresponding eigenvectors (row-normalized)
        - sorted_indices: indices of sorted eigenvalues
    """
    W = model.linear.weight.detach()  # shape: (out_dim, input_dim)
    weight_norms = torch.norm(W, dim=1)  # ||w_i|| for each row
    lambda_estimates = 1.0 / (weight_norms ** 2)

    # Normalize rows of W to get eigenvector directions
    eigvecs = F.normalize(W, p=2, dim=1)  # shape: (out_dim, input_dim)

    # Sort eigenvalues and reorder eigenvectors
    eigvals_sorted, indices = torch.sort(lambda_estimates, descending=True)
    eigvecs_sorted = eigvecs[indices]

    return eigvals_sorted, eigvecs_sorted, indices

def compare_eigenvectors_cosine(eigvecs_learned, eigvecs_true):
    """
    Computes cosine similarity between corresponding eigenvectors
    from the learned and true (sorted) bases.

    Assumes both inputs are tensors of shape (num_vectors, dim),
    with eigenvectors as rows.

    Returns:
        Tensor of cosine similarities, one per matched pair.
    """
    assert eigvecs_learned.shape == eigvecs_true.shape, \
        "Eigenvector sets must have the same shape"

    cos_sims = F.cosine_similarity(eigvecs_learned, eigvecs_true, dim=1)
    return cos_sims

def match_eigenvectors_optimal(learned_vecs, true_vecs, return_matrix=True):
    """
    Compares all pairs of learned and true eigenvectors using cosine similarity.
    Finds the optimal 1-to-1 assignment using the Hungarian algorithm.

    Args:
        learned_vecs: Tensor of shape (k, N)
        true_vecs: Tensor of shape (k, N)

    Returns:
        match_indices: list of matched indices (learned i → true match[i])
        similarities: list of cosine similarities for the matches
        (optional) sim_matrix: full cosine similarity matrix
    """
    # Compute cosine similarity matrix
    sim_matrix = F.cosine_similarity(
        learned_vecs.unsqueeze(1),  # (k, 1, N)
        true_vecs.unsqueeze(0),     # (1, k, N)
        dim=2                       # → (k, k)
    ).cpu().numpy()

    # Convert to cost matrix for maximization (negate)
    cost_matrix = -np.abs(sim_matrix)

    # Hungarian algorithm: minimizes cost
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    matched_sims = sim_matrix[row_ind, col_ind]

    if return_matrix:
        return col_ind, matched_sims, sim_matrix
    else:
        return col_ind, matched_sims

def test_whitening(W, cov, label="W", verbose=True):
    """
    Test whether W is a whitening basis for the covariance matrix `cov`.
    Computes W @ cov @ W.T and compares it to identity.
    
    Args:
        W (torch.Tensor): Shape (k, N) — rows are basis vectors.
        cov (torch.Tensor): Shape (N, N) — original covariance matrix.
        label (str): Name for the test (e.g., "ReLU", "Abs").
        verbose (bool): Whether to print detailed metrics.
        
    Returns:
        fro_norm (float): Frobenius norm of (WΣWᵀ - I)
    """
    W = F.normalize(W, dim=1)  # ensure rows are unit vectors
    whitened_cov = W @ cov @ W.T
    k = W.shape[0]
    identity = torch.eye(k, device=W.device)
    diff = whitened_cov - identity
    fro_norm = torch.norm(diff, p='fro').item()

    if verbose:
        eigvals = torch.linalg.eigvalsh(whitened_cov).cpu().numpy()
        print(f"[{label}] Whitening Test:")
        print(f"  Frobenius norm ||WΣWᵀ - I||_F = {fro_norm:.6f}")
        print(f"  Eigenvalues of WΣWᵀ: {np.round(eigvals, 4)}")

    return fro_norm

def test_sphericity(W, cov, label, verbose=True):
    """
    Test whether the learned weights transform the input covariance structure into a spherical distribution.
    
    Args:
        model: The neural network model with a linear layer
        cov: The covariance matrix of the input distribution
    
    Returns:
        sphericity_error: Frobenius norm measuring deviation from perfect sphericity
        eigenvalues: Eigenvalues of the transformed covariance matrix (normalized)
    """
    # Compute the transformed covariance matrix WΣW^T
    transformed_cov = W @ cov @ W.T
    
    # Calculate the mean of the diagonal elements (average variance along each dimension)
    diag_mean = torch.mean(torch.diag(transformed_cov))

    # Normalize by the mean diagonal to measure sphericity, not just whitening
    # A perfectly spherical distribution would have all eigenvalues equal after this normalization
    normalized_cov = transformed_cov / diag_mean
    
    identity = torch.eye(normalized_cov.shape[0], device=W.device)

    # Compute deviation from the identity matrix as fro_norm (perfect sphericity)
    sphericity_error = torch.norm(normalized_cov - identity, 'fro')
    
    # Compute eigenvalues for further analysis
    eigenvalues = torch.linalg.eigvalsh(normalized_cov)
    
    if verbose:
        print(f"[{label}] Sphericity Test:")
        print(f"  Frobenius norm ||WΣWᵀ/mean(diag(WΣWᵀ)) - I||_F = {sphericity_error:.6f}")
        print(f"  Eigenvalues of normalized WΣWᵀ: {eigenvalues.tolist()}")
    
    return sphericity_error, eigenvalues

def detect_mirrored_weights(W, cos_threshold=0.999):
    """
    Detects near-opposite vector pairs (w, -w) in a weight matrix W.

    Args:
        W (Tensor): shape (n, d) where each row is a weight vector.
        cos_threshold (float): cosine similarity threshold for "mirror-ness".

    Returns:
        mirrored_pairs (List[Tuple[int, int, float]]): 
            list of (i, j, cosine_similarity) where w_i ≈ -w_j.
        used_indices (Set[int]): all indices involved in mirrored pairs.
    """
    W = F.normalize(W, dim=1)
    n = W.shape[0]
    mirrored_pairs = []
    used = set()

    for i in range(n):
        if i in used:
            continue
        wi = W[i]
        for j in range(i + 1, n):
            if j in used:
                continue
            wj = W[j]
            cos_sim = F.cosine_similarity(wi.unsqueeze(0), wj.unsqueeze(0)).item()
            if cos_sim < -cos_threshold:  # close to -1
                mirrored_pairs.append((i, j, cos_sim))
                used.update([i, j])
                break  # only match each i once

    return mirrored_pairs, used

def deduplicate_mirrored_weights(W, mirrored_pairs):
    """
    Deduplicates mirrored vectors in W by keeping one from each mirrored pair,
    and including all unpaired vectors.

    Args:
        W (Tensor): (n, d) matrix of weight vectors.
        mirrored_pairs (List[Tuple[int, int, float]]): Output from detect_mirrored_weights.

    Returns:
        deduped_W (Tensor): Reduced set of weight vectors with no mirrored redundancy.
    """
    import torch

    # Indices from mirrored pairs (keep first index from each)
    keep_indices = set(i for (i, _, _) in mirrored_pairs)

    # Indices of all used vectors in mirrored pairs
    mirrored_indices = set(i for (i, j, _) in mirrored_pairs).union(j for (_, j, _) in mirrored_pairs)

    # Add any unused vectors (i.e., not part of any mirrored pair)
    all_indices = set(range(W.shape[0]))
    unused_indices = all_indices - mirrored_indices
    keep_indices.update(unused_indices)

    # Sort for consistency (optional)
    keep_indices = sorted(keep_indices)

    return W[keep_indices]

def print_mirrored_summary(name, mirrored_pairs, total):
    count = len(mirrored_pairs)
    print(f"{name:<8} | {count:>2} mirrored pairs out of {total:>2} vectors")
    if count > 0:
        print(f"{'Idx A':>6} {'Idx B':>6} {'cos(wᵢ, wⱼ)':>12}")
        print("-" * 26)
        for i, j, sim in mirrored_pairs:
            print(f"{i:6d} {j:6d} {sim:12.6f}")

def mahalanobis_distance(x, mean, eigvals_sorted, eigvecs_sorted):
    centered = x - mean  # shape: (num_samples, N)
    proj = centered @ eigvecs_sorted  # shape: (num_samples, N)
    scaled = proj / torch.sqrt(eigvals_sorted)  # element-wise division, shape: (num_samples, N)
    md = torch.norm(scaled, dim=1)  # Mahalanobis distance for each sample
    return md

def summarize_trials(results_dir="results/learn_md", model_types=("relu", "abs", "sigmoid")):
    trial_dirs = [os.path.join(results_dir, d) for d in os.listdir(results_dir)
                  if os.path.isdir(os.path.join(results_dir, d)) and d.isdigit()]
    trial_dirs.sort(key=lambda x: int(os.path.basename(x)))

    summary = {m: [] for m in model_types}
    best_models = {}

    for trial_path in trial_dirs:
        for model_type in model_types:
            stats_path = os.path.join(trial_path, f"{model_type}_stats.pt")
            if not os.path.exists(stats_path):
                continue

            losses, stats = torch.load(stats_path)
            final_loss = losses[-1]

            summary[model_type].append({
                "trial": trial_path,
                "losses": losses,
                "final_loss": final_loss,
                "stats": stats,
            })

            # Update best model if this one is better
            if model_type not in best_models or final_loss < best_models[model_type]["final_loss"]:
                best_models[model_type] = {
                    "trial": trial_path,
                    "final_loss": final_loss,
                    "model_path": os.path.join(trial_path, f"{model_type}_model.pt"),
                    "stats_path": stats_path,
                }

    # Print summary statistics
    for model_type in model_types:
        losses = [entry["final_loss"] for entry in summary[model_type]]
        if losses:
            mean = np.mean(losses)
            var = np.var(losses)
            min_loss = np.min(losses)
            max_loss = np.max(losses)
            print(f"{model_type.upper()} - Mean: {mean:.6f}, Var: {var:.6f}, Min: {min_loss:.6f}, Max: {max_loss:.6f}")

    # Copy best models to results/learn_md
    for model_type, info in best_models.items():
        shutil.copy(info["model_path"], os.path.join(results_dir, f"{model_type}_model.pt"))
        shutil.copy(info["stats_path"], os.path.join(results_dir, f"{model_type}_stats.pt"))
        print(f"Best {model_type} model copied from {info['trial']}")

    return summary, best_models

def eval_model(model, x, y, label):
    model.eval()
    y_pred = model(x)
    loss = F.mse_loss(y_pred, y).item()
    print(f"{label} model loss on new data: {loss:.6f}")

def plot_2d_data(x):
    plt.figure(figsize=(6, 6))
    plt.scatter(x[:, 0].cpu(), x[:, 1].cpu(), s=10, alpha=0.5)
    plt.title("Top Gaussian Components Data")
    plt.axis('equal')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_model_hyperplanes(x, mean, model, title="Model Hyperplanes", filename=None):
    x = x.detach().cpu()
    mean = mean.detach()
    W = model.linear.weight.detach()
    b = model.linear.bias.detach()
    
    plt.figure(figsize=(6, 6))

    data_color = '#666666'        # Lighter gray
    hyperplane_color = '#333333'  # Darker gray
    arrow_color = '#333333'       # Darker gray
    scale_factor = 5

    # Main scatter plot
    plt.scatter(x[:, 0], x[:, 1], s=10, alpha=0.3, color=data_color, edgecolors='none', label="Data")

    for i in range(W.shape[0]):
        w = W[i]
        norm = torch.norm(w)
        if norm == 0:
            continue
        normal = w / norm

        # Distance from mean to hyperplane along the normal
        distance = (w @ mean + b[i]) / norm

        # Project the mean onto the hyperplane
        projection_on_plane = mean - distance * normal

        # Use perpendicular direction for the hyperplane line
        perp = torch.tensor([-normal[1], normal[0]], device=normal.device)
        line_center = projection_on_plane

        # Extend the line
        scale_factor = 5
        pt1 = line_center + perp * scale_factor
        pt2 = line_center - perp * scale_factor

        # Convert to CPU
        pt1 = pt1.cpu()
        pt2 = pt2.cpu()
        normal = normal.cpu()
        proj = projection_on_plane.cpu()

        # Plot the hyperplane
        plt.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], 
        color=hyperplane_color, linewidth=1.5, linestyle='--')

        # Normal vector
        arrow_scale = 1.0 / torch.sqrt(norm).item()

        plt.arrow(
            proj[0].item(), proj[1].item(),
            normal[0].item() * arrow_scale * 2, normal[1].item() * arrow_scale * 2,
            head_width=0.25, head_length=0.35,
            fc=arrow_color, ec=arrow_color,
            length_includes_head=True, width=0.05, zorder=3
        )

    plt.title(title, fontsize=14, weight='bold')
    plt.xlabel("x0", fontsize=12)
    plt.ylabel("x1", fontsize=12)

    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    plt.axis('equal')

    # Ticks
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)

    plt.tight_layout()

    if filename:
        directory = os.path.dirname(filename)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)

        plt.savefig(filename)
        print(f"Saved: {filename}")
    else:
        plt.show()

def make_ground_truth_model(mean, eigvecs, eigvals):
    """
    Constructs an MD_Abs model that computes Mahalanobis distance using PCA components.
    
    Args:
        mean (Tensor): Mean vector (shape: [d])
        eigvecs (Tensor): Matrix of eigenvectors (shape: [d, d], column-wise)
        eigvals (Tensor): Vector of eigenvalues (shape: [d])
    
    Returns:
        MD_Abs: A model initialized to compute Mahalanobis distance from mean and covariance.
    """
    input_dim = mean.shape[0]
    
    # Whitening: W = diag(1 / sqrt(λ)) @ Vᵀ
    W = torch.diag(1.0 / torch.sqrt(eigvals)) @ eigvecs.T  # shape: [d, d]
    
    # Bias: b = -W @ μ
    b = -W @ mean  # shape: [d]
    
    # Create the model
    model = MD_Abs(input_dim, weight=W, bias=b)
    return model
