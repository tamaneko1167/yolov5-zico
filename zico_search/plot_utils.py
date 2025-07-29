# zico_search/plot
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

PLOT_DIR = os.path.join(os.path.dirname(__file__), "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

def plot_param_vs_score(params, scores, param_name, filename):
    plt.figure()
    plt.scatter(params, scores, marker="o")
    plt.xlabel(param_name)
    plt.ylabel("ZiCo Score")
    plt.title(f"{param_name} vs ZiCo Score")
    plt.grid(True)
    plt.tight_layout()

    save_path = os.path.join(PLOT_DIR, filename)
    plt.savefig(save_path)
    print(f"[Saved] Plot saved to: {save_path}")
    plt.close()

def plot_depth_flops_params(results_with_hyperparams, save_path):
    depth_list = [r[3] for r in results_with_hyperparams]
    flops_list = [r[2] for r in results_with_hyperparams]
    params_list = [r[1] for r in results_with_hyperparams]
    zico_scores = [r[0] for r in results_with_hyperparams]

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    scatter = ax.scatter(flops_list, params_list, zico_scores, c=depth_list, cmap='plasma', s=40)
    fig.colorbar(scatter, label="Depth Multiple")

    ax.set_xlabel("FLOPs")
    ax.set_ylabel("Params")
    ax.set_zlabel("ZiCo Score")
    plt.title("FLOPs × Params (Depth Score color)")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_param_vs_metric(results, param_index, metric_index, xlabel, ylabel, save_path):
    """
    Plot a given parameter (depth or width) vs. a specific metric (ZiCo score, FLOPs, or Params).

    Args:
        results: List of tuples (zico, params, flops, depth, width)
        param_index: Index of the parameter to plot on x-axis (3=depth, 4=width)
        metric_index: Index of the metric to plot on y-axis (0=ZiCo, 1=Params, 2=FLOPs)
        xlabel: Label for x-axis (e.g., "Depth Multiple" or "Width Multiple")
        ylabel: Label for y-axis (e.g., "ZiCo Score")
        save_path: Path to save the output plot
    """
    x_values = [r[param_index] for r in results]
    y_values = [r[metric_index] for r in results]

    plt.figure(figsize=(8, 6))
    plt.scatter(x_values, y_values, c='blue', alpha=0.7)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(f"{xlabel} vs. {ylabel}")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_importances(importances_dict, save_path):
    # importances_dict: {'param_name': importance, ...}
    names = list(importances_dict.keys())
    vals = [float(importances_dict[k]) for k in names]
    plt.figure(figsize=(6, 3))
    plt.barh(names, vals)
    plt.xlabel("Importance")
    plt.title("Hyperparameter Importance")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
