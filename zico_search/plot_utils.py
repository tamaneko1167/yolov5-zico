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
