import analyze_md
import torch
import learn_md

from models_md import MD_ReLU, MD_Abs, MD_Sigmoid
from pathlib import Path

results_dir = Path("results/md_2d/")
models_dir = results_dir / "models"
hyperplane_dir = results_dir / "hyperplane_plots"
hyperplane_dir.mkdir(parents=True, exist_ok=True)

analyze_md.summarize_trials(results_dir=models_dir)

N, mean, cov, eigvals_sorted, eigvecs_sorted = torch.load(results_dir / "data_model.pt")

rlu_model = analyze_md.load_model(MD_ReLU, N, models_dir / "relu_model.pt")
abs_model = analyze_md.load_model(MD_Abs, N, models_dir / "abs_model.pt")
sig_model = analyze_md.load_model(MD_Sigmoid, N, models_dir / "sigmoid_model.pt")

rlu_eigvals, rlu_eigvecs, rlu_eigsort = analyze_md.recover_sorted_eigenstructure(rlu_model)
abs_eigvals, abs_eigvecs, abs_eigsort = analyze_md.recover_sorted_eigenstructure(abs_model)
sig_eigvals, sig_eigvecs, sig_eigsort = analyze_md.recover_sorted_eigenstructure(sig_model)

rlu_fro_norm = analyze_md.test_whitening(rlu_model.linear.weight.detach(), cov)
abs_fro_norm = analyze_md.test_whitening(abs_model.linear.weight.detach(), cov)
sig_fro_norm = analyze_md.test_whitening(sig_model.linear.weight.detach(), cov)

print(f"Distance to Mean")
print(f"\t ReLU    {analyze_md.distance_from_mean_to_hyperplanes(rlu_model, mean)}")
print(f"\t Abs     {analyze_md.distance_from_mean_to_hyperplanes(abs_model, mean)}")
print(f"\t Sigmoid {analyze_md.distance_from_mean_to_hyperplanes(sig_model, mean)}")

print(f"Eigenvalues from weights")
print(f"\t ReLU    {rlu_eigvals}")
print(f"\t Abs     {abs_eigvals}")
print(f"\t Sigmoid {sig_eigvals}")
print(f"\t Actual  {eigvals_sorted}")

print(f"Eigenvectors from weights")
print(f"\t ReLU    {rlu_eigvecs}")
print(f"\t Abs     {abs_eigvecs}")
print(f"\t Sigmoid {sig_eigvecs}")
print(f"\t Actual  {eigvecs_sorted}")

print(f"Whitening Frobius norms")
print(f"\t ReLU    {rlu_fro_norm}")
print(f"\t Abs     {abs_fro_norm}")
print(f"\t Sigmoid {sig_fro_norm}")

print(f"Mahalanbois Frobius norms")
print(f"\t ReLU    {analyze_md.test_mahalanobis_match(rlu_model.linear.weight.detach(), cov)}")
print(f"\t Abs     {analyze_md.test_mahalanobis_match(abs_model.linear.weight.detach(), cov)}")
print(f"\t Sigmoid {analyze_md.test_mahalanobis_match(sig_model.linear.weight.detach(), cov)}")

num_samples = 1000
x = torch.distributions.MultivariateNormal(mean, covariance_matrix=cov).sample((num_samples,))
y = learn_md.mahalanobis_distance(x, mean, eigvals_sorted, eigvecs_sorted)

analyze_md.eval_model(rlu_model, x, y)
analyze_md.eval_model(abs_model, x, y)
analyze_md.eval_model(sig_model, x, y)

analyze_md.plot_model_hyperplanes(x, mean, rlu_model, "ReLU Decision Boundaries", hyperplane_dir / "relu_boundaries.png")
analyze_md.plot_model_hyperplanes(x, mean, abs_model, "Abs Decision Boundaries", hyperplane_dir / "abs_boundaries.png")
analyze_md.plot_model_hyperplanes(x, mean, sig_model, "Sigmoid Decision Boundaries", hyperplane_dir / "sigmoid_boundaries.png")
