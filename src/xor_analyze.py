# xor_analyze.py

import torch
import xor_lib
import matplotlib.pyplot as plt
from pathlib import Path
from xor_models import XOR_ReLU, XOR_Abs, XOR_Sigmoid
import matplotlib as mpl

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

results_dir = Path("results/xor/")
models_dir = results_dir / "models"
plots_dir = results_dir / "hyperplane_plots"
plots_dir.mkdir(parents=True, exist_ok=True)

# Load XOR dataset
x, y = xor_lib.create_xor_dataset()
x = x.to(device)
y = y.to(device)

# Model classes
model_classes = {
    "relu": XOR_ReLU,
    "abs": XOR_Abs,
    "sigmoid": XOR_Sigmoid
}

# Accuracy counters
accuracy_counts = {
    "relu": {4: 0, 3: 0, 2: 0, 1: 0, 0: 0},
    "abs": {4: 0, 3: 0, 2: 0, 1: 0, 0: 0},
    "sigmoid": {4: 0, 3: 0, 2: 0, 1: 0, 0: 0}
}

# Best models (first perfect one found)
best_models = {}

# Track best trial index
best_model_indices = {}

# Analyze each trial
trial_dirs = sorted([d for d in models_dir.iterdir() if d.is_dir() and d.name.isdigit()],
                    key=lambda d: int(d.name))

def plot_model(model, title, filename):
    mpl.rcParams['font.family'] = 'DejaVu Sans'  # or 'Arial'
    mpl.rcParams['axes.titlesize'] = 16
    mpl.rcParams['axes.labelsize'] = 14
    mpl.rcParams['xtick.labelsize'] = 12
    mpl.rcParams['ytick.labelsize'] = 12
    mpl.rcParams['legend.fontsize'] = 12

    model = model.eval()

    x_cpu = x.detach().cpu()
    y_cpu = y.detach().cpu()

    W = None
    b = None

    if isinstance(model, XOR_ReLU) or isinstance(model, XOR_Sigmoid):
        W = model.linear1.weight.detach()
        b = model.linear1.bias.detach()
    elif isinstance(model, XOR_Abs):
        W = model.linear1.weight.detach()
        b = model.linear1.bias.detach()

    W = W.to('cpu')
    b = b.to('cpu')

    mean = torch.zeros(2)  # For XOR, centered at (0,0)

    plt.figure(figsize=(6, 6))

    data_color = '#666666'        # Light gray
    hyperplane_color = '#333333'  # Darker gray
    arrow_color = '#333333'       # Darker gray
    scale_factor = 5

    # Scatter plot for XOR points
    # colors = ['red' if label == 0 else 'blue' for label in y_cpu]
    # plt.scatter(x_cpu[:, 0], x_cpu[:, 1], s=100, color=colors, edgecolors='k', label="Data")
    # Plot data points with different markers
    for xi, yi in zip(x_cpu, y_cpu):
        if yi == 0:
            plt.scatter(xi[0], xi[1], marker='o', s=100, color='black', edgecolors='k', linewidths=1)
        else:
            plt.scatter(xi[0], xi[1], marker='^', s=100, color='black', edgecolors='k', linewidths=1)

    for i in range(W.shape[0]):
        w = W[i]
        norm = torch.norm(w)
        if norm == 0:
            continue
        normal = w / norm

        # Distance from mean to hyperplane
        distance = (w @ mean + b[i]) / norm

        # Project mean onto hyperplane
        projection_on_plane = mean - distance * normal

        perp = torch.tensor([-normal[1], normal[0]])  # Perpendicular direction
        pt1 = projection_on_plane + perp * scale_factor
        pt2 = projection_on_plane - perp * scale_factor

        # Plot hyperplane
        plt.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]],
                 color=hyperplane_color, linewidth=1.5, linestyle='--')

        arrow_length = 0.5  # fixed shorter arrow
        plt.arrow(
            projection_on_plane[0].item(), projection_on_plane[1].item(),
            normal[0].item() * arrow_length, normal[1].item() * arrow_length,
            head_width=0.15, head_length=0.2,
            fc=arrow_color, ec=arrow_color,
            alpha=1.0,  # transparency
            length_includes_head=True, width=0.03, zorder=3
        )

    plt.title(title, fontsize=16, weight='bold', pad=12)
    plt.xlabel("x0", fontsize=12)
    plt.ylabel("x1", fontsize=12)

    plt.xlim(-2.2, 2.2)
    plt.ylim(-2.2, 2.2)

    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)
    plt.axis('equal')
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)

    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)

    plt.tight_layout()

    if filename:
        plt.savefig(filename)
        # print(f"Saved plot: {filename}")
    else:
        plt.show()

    plt.close()

def detect_mirrored_relu(model):
    W = model.linear1.weight.detach()
    W = W / W.norm(dim=1, keepdim=True)
    cos_sim = torch.matmul(W, W.t())
    return cos_sim

for trial_dir in trial_dirs:
    trial_index = int(trial_dir.name)

    for model_type in model_classes:
        model_path = trial_dir / f"{model_type}_model.pt"
        if not model_path.exists():
            print(f"Missing model: {model_path}")
            continue

        # Load model
        model = model_classes[model_type]().to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

        # Evaluate
        correct = xor_lib.evaluate_model(model, x, y)
        accuracy_counts[model_type][correct] += 1
        if model_type == "relu":
            # Normalize weights
            W = model.linear1.weight.detach()
            W = W / W.norm(dim=1, keepdim=True)
            cos_sim = torch.matmul(W, W.t())
            print(f"\nReLU Weight Cosine Similarities for trial {trial_dir.name}:")
            print(cos_sim)
            print(f"Accuracy: {correct}/4")

        # Save first perfect model
        if correct == 4 and model_type not in best_models:
            best_models[model_type] = (model, trial_dir)
            best_model_indices[model_type] = trial_index

        # Plot each model individually into its trial folder
        plot_filename = trial_dir / f"{model_type}_boundaries.png"
        plot_model(model, f"{model_type.upper()} Trial {trial_index}", plot_filename)

# Print summary
print("\n=== Accuracy Summary ===")
for model_type in accuracy_counts:
    print(f"\n{model_type.upper()}:")
    for num_correct, count in sorted(accuracy_counts[model_type].items(), reverse=True):
        print(f"  {num_correct}/4 correct: {count} runs")

# Plot best models separately into hyperplane_plots/
for model_type, (model, trial_dir) in best_models.items():
    plot_model(model, f"{model_type.upper()} Best Model", plots_dir / f"{model_type}_boundaries.png")
    print(f"Best {model_type.upper()} model found at trial index: {best_model_indices[model_type]}")
