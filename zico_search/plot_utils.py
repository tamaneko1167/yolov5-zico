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


import numpy as np
from scipy.interpolate import make_interp_spline
    
# def plot_param_vs_score(xs, ys, param_name="param", filename="plot.png"):
#     plt.figure(figsize=(8, 5))

#     # Step 1: 外れ値除去（IQRを使う）
#     ys_np = np.array(ys)
#     q1 = np.percentile(ys_np, 25)
#     q3 = np.percentile(ys_np, 75)
#     iqr = q3 - q1
#     lower_bound = q1 - 1.5 * iqr
#     upper_bound = q3 + 1.5 * iqr

#     filtered = [(x, y) for x, y in zip(xs, ys) if lower_bound <= y <= upper_bound]
#     if len(filtered) < 4:
#         print("⚠️ 外れ値除去後のデータが少なすぎます。全データを使用します。")
#         filtered = list(zip(xs, ys))

#     xs_filtered, ys_filtered = zip(*filtered)

#     # Step 2: 散布図
#     plt.scatter(xs_filtered, ys_filtered, label="Trials", color="blue", alpha=0.6)

#     # Step 3: スムージング
#     if len(xs_filtered) >= 4:
#         xs_sorted, ys_sorted = zip(*sorted(zip(xs_filtered, ys_filtered)))
#         xs_new = np.linspace(min(xs_sorted), max(xs_sorted), 300)
#         spline = make_interp_spline(xs_sorted, ys_sorted, k=3)
#         ys_smooth = spline(xs_new)
#         plt.plot(xs_new, ys_smooth, label="Smoothed", color="red")

#     # Y軸の表示範囲を拡大表示にする
#     margin = 0.2 * (max(ys_filtered) - min(ys_filtered))
#     plt.ylim(min(ys_filtered) - margin, max(ys_filtered) + margin)

#     plt.xlabel(param_name)
#     plt.ylabel("ZiCo Score")
#     plt.title(f"{param_name} vs ZiCo Score")
#     plt.legend()
#     plt.grid(True)
#     plt.tight_layout()
#     save_path = os.path.join(PLOT_DIR, filename)
#     plt.savefig(save_path)
#     print(f"[Saved] {filename}")
