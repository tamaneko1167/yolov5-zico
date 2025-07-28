# zico_search/plot
import os
import matplotlib.pyplot as plt

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

def plot_width_vs_metric(results, metric_index, ylabel, save_path):
    """
    Plot width_multiple vs. a specific metric (ZiCo score, FLOPs, or Params)
    """
    widths = [r[4] for r in results]
    metrics = [r[metric_index] for r in results]

    plt.figure(figsize=(8, 6))
    plt.scatter(widths, metrics, c='blue', alpha=0.7)
    plt.xlabel("Width Multiple")
    plt.ylabel(ylabel)
    plt.title(f"Width Multiple vs. {ylabel}")
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
