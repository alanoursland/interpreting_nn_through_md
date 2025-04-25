import analyze_md
import torch
import learn_md

from md_models import MD_ReLU, MD_Abs, MD_Sigmoid
from pathlib import Path
from collections import defaultdict

results_dir = Path("results/md_2d/")
models_dir = results_dir / "models"
hyperplane_dir = results_dir / "hyperplane_plots"
hyperplane_dir.mkdir(parents=True, exist_ok=True)

print("Finding best models:")
analyze_md.summarize_trials(results_dir=models_dir)
print()

N, mean, cov, eigvals_sorted, eigvecs_sorted = torch.load(results_dir / "data_model.pt")

def analyze_best_models():
    rlu_model = analyze_md.load_model(MD_ReLU, N, models_dir / "relu_model.pt")
    abs_model = analyze_md.load_model(MD_Abs, N, models_dir / "abs_model.pt")
    sig_model = analyze_md.load_model(MD_Sigmoid, N, models_dir / "sigmoid_model.pt")
    gnd_model = analyze_md.make_ground_truth_model(mean, eigvecs_sorted, eigvals_sorted).to(cov.device)

    rlu_W = rlu_model.linear.weight.detach()
    abs_W = abs_model.linear.weight.detach()
    sig_W = sig_model.linear.weight.detach()
    gnd_W = gnd_model.linear.weight.detach()

    rlu_mirrored_pairs, rlu_mirrored_set = analyze_md.detect_mirrored_weights(rlu_W)
    abs_mirrored_pairs, abs_mirrored_set = analyze_md.detect_mirrored_weights(abs_W)
    sig_mirrored_pairs, sig_mirrored_set = analyze_md.detect_mirrored_weights(sig_W)
    gnd_mirrored_pairs, gnd_mirrored_set = analyze_md.detect_mirrored_weights(gnd_W)

    relu_W_mirror = analyze_md.deduplicate_mirrored_weights(rlu_W, rlu_mirrored_pairs)

    rlu_mean_dist = analyze_md.distance_from_mean_to_hyperplanes(rlu_model, mean)
    abs_mean_dist = analyze_md.distance_from_mean_to_hyperplanes(abs_model, mean)
    sig_mean_dist = analyze_md.distance_from_mean_to_hyperplanes(sig_model, mean)
    gnd_mean_dist = analyze_md.distance_from_mean_to_hyperplanes(gnd_model, mean)

    rlu_eigvals, rlu_eigvecs, rlu_eigsort = analyze_md.recover_sorted_eigenstructure(rlu_model)
    abs_eigvals, abs_eigvecs, abs_eigsort = analyze_md.recover_sorted_eigenstructure(abs_model)
    sig_eigvals, sig_eigvecs, sig_eigsort = analyze_md.recover_sorted_eigenstructure(sig_model)
    gnd_eigvals, gnd_eigvecs, gnd_eigsort = analyze_md.recover_sorted_eigenstructure(gnd_model)

    rlu_sphericity, _ = analyze_md.test_sphericity(relu_W_mirror, cov, label="ReLU", verbose=False)
    abs_sphericity, _ = analyze_md.test_sphericity(abs_W, cov, label="Abs", verbose=False)
    sig_sphericity, _ = analyze_md.test_sphericity(sig_W, cov, label="Sigmoid", verbose=False)
    gnd_sphericity, _ = analyze_md.test_sphericity(gnd_W, cov, label="Ground", verbose=False)

    print("Mirrored Weight Detection Summary")
    print(f"\t ReLU    {len(rlu_mirrored_pairs)} out of {rlu_W.shape[0]} vectors")
    print(f"\t Abs     {len(abs_mirrored_pairs)} out of {abs_W.shape[0]} vectors")
    print(f"\t Sigmoid {len(sig_mirrored_pairs)} out of {sig_W.shape[0]} vectors")
    print(f"\t Ground  {len(gnd_mirrored_pairs)} out of {gnd_W.shape[0]} vectors")
    print()

    print(f"Distance to Mean")
    print(f"\t ReLU    {rlu_mean_dist}")
    print(f"\t Abs     {abs_mean_dist}")
    print(f"\t Sigmoid {sig_mean_dist}")
    print(f"\t Ground  {gnd_mean_dist}")
    print()

    print(f"Sphericity Frobius norms")
    print(f"\t ReLU    {rlu_sphericity}")
    print(f"\t Abs     {abs_sphericity}")
    print(f"\t Sigmoid {sig_sphericity}")
    print(f"\t Ground  {gnd_sphericity}")
    print()

    print(f"Eigenvalues from weights")
    print(f"\t ReLU    {rlu_eigvals}")
    print(f"\t Abs     {abs_eigvals}")
    print(f"\t Sigmoid {sig_eigvals}")
    print(f"\t Ground  {gnd_eigvals}")
    print(f"\t Actual  {eigvals_sorted}")
    print()

    print(f"Eigenvectors from weights")
    print(f"\t ReLU    {rlu_eigvecs}")
    print(f"\t Abs     {abs_eigvecs}")
    print(f"\t Sigmoid {sig_eigvecs}")
    print(f"\t Ground  {gnd_eigvecs}")
    print(f"\t Actual  {eigvecs_sorted}")
    print()

    num_samples = 1000
    x = torch.distributions.MultivariateNormal(mean, covariance_matrix=cov).sample((num_samples,))
    y = learn_md.mahalanobis_distance(x, mean, eigvals_sorted, eigvecs_sorted)

    analyze_md.eval_model(rlu_model, x, y, label="ReLU")
    analyze_md.eval_model(abs_model, x, y, label="Abs")
    analyze_md.eval_model(sig_model, x, y, label="Sigmoid")
    analyze_md.eval_model(gnd_model, x, y, label="Ground")
    print()

    analyze_md.plot_model_hyperplanes(x, mean, gnd_model, "Ground Truth Decision Boundaries", hyperplane_dir / "ground_boundaries.png")
    analyze_md.plot_model_hyperplanes(x, mean, rlu_model, "ReLU Decision Boundaries", hyperplane_dir / "relu_boundaries.png")
    analyze_md.plot_model_hyperplanes(x, mean, abs_model, "Abs Decision Boundaries", hyperplane_dir / "abs_boundaries.png")
    analyze_md.plot_model_hyperplanes(x, mean, sig_model, "Sigmoid Decision Boundaries", hyperplane_dir / "sigmoid_boundaries.png")

analyze_best_models()
print()

def analyze_all_model_runs(results_dir, model_types, mean, cov):
    """
    Analyze representation metrics across all model runs for each architecture.
    
    Args:
        results_dir (str or Path): Path to the directory with numbered trial subfolders.
        model_types (tuple[str]): Model type names (e.g., 'relu', 'abs', 'sigmoid').
        mean (Tensor): Mean of the Gaussian distribution.
        cov (Tensor): Covariance matrix of the Gaussian distribution.
    """
    model_classes = {
        "abs": MD_Abs,
        "relu": MD_ReLU,
        "sigmoid": MD_Sigmoid
    }

    results_dir = Path(results_dir)
    trial_dirs = sorted([d for d in results_dir.iterdir() if d.is_dir() and d.name.isdigit()],
                        key=lambda d: int(d.name))
    
    all_stats = defaultdict(lambda: defaultdict(list))

    for trial_dir in trial_dirs:
        for model_type in model_types:
            model_path = trial_dir / f"{model_type}_model.pt"
            if not model_path.exists():
                print(f"Cannot find model at {model_path}")
                continue

            try:
                model = analyze_md.load_model(model_classes[model_type], mean.shape[0], model_path)
            except Exception as e:
                print(f"Failed to load {model_path}: {e}")
                continue

            W = model.linear.weight.detach()
            b = model.linear.bias.detach()

            if model_type == "relu":
                mirrored_pairs, _ = analyze_md.detect_mirrored_weights(W)
                mirrored_count = len(mirrored_pairs)
                W_proc = analyze_md.deduplicate_mirrored_weights(W, mirrored_pairs)
            else:
                mirrored_count = 0
                W_proc = W

            mean_dist = analyze_md.distance_from_mean_to_hyperplanes(model, mean).mean().item()
            sphericity, _ = analyze_md.test_sphericity(W_proc, cov, label=None, verbose=False)

            all_stats[model_type]["mirrored_count"].append(mirrored_count)
            all_stats[model_type]["mean_dist"].append(mean_dist)
            all_stats[model_type]["sphericity"].append(sphericity)

    # Print summary
    for model_type in model_types:
        print(f"{model_type.upper()} INTERPRETABILITY SUMMARY")
        for metric, values in all_stats[model_type].items():
            # Convert values to a PyTorch tensor
            values_tensor = torch.tensor(values, dtype=torch.float32)
            print(f"  {metric.replace('_', ' ').capitalize()}: "
                f"Mean = {values_tensor.mean().item():.6f}, Std = {values_tensor.std().item():.6f}, "
                f"Min = {values_tensor.min().item():.6f}, Max = {values_tensor.max().item():.6f}")

    return all_stats

print("Analysis across runs:")
interpretability = analyze_all_model_runs(models_dir, model_types = ("relu", "abs", "sigmoid"), mean=mean, cov=cov)
# print(interpretability)

