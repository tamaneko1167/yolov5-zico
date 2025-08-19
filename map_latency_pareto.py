import numpy as np
import matplotlib.pyplot as plt

# ====== サンプルデータ ======
# (depth, sppf, gamma, mAP, fps, latency_ms)
data = [
    (0.33, True,  1.0, 0.4830, 1.75, 572.48),  # baseline
    (0.33, False,  1.5, 0.3988, 2.05, 488.26),  # best
    (0.33, True, 1.5, 0.4842, 1.61, 621.28),
    (0.76, False,  1.0, 0.4874, 1.60, 624.42),
]

depth   = np.array([d[0] for d in data])
sppf    = np.array([d[1] for d in data])
gamma   = np.array([d[2] for d in data])
mAP     = np.array([d[3] for d in data])
fps     = np.array([d[4] for d in data])
lat_ms  = np.array([d[5] for d in data])  # ms

# ====== baseline と best を指定 ======
baseline_idx = 0
best_idx = 1

def plot_graph(x, x_label, title):
    plt.figure(figsize=(8,6))

    # 全点は青で散布
    plt.scatter(x, mAP, alpha=0.5, color="blue", label="All configs")

    # baseline を赤で強調
    plt.scatter(x[baseline_idx], mAP[baseline_idx], color="red", s=100, label="Baseline")

    # best を緑で強調
    plt.scatter(x[best_idx], mAP[best_idx], color="green", s=100, label="Best")

    # ====== 全部にラベル表示 ======
    for i in range(len(data)):
        lbl = f"d={depth[i]}, sppf={sppf[i]}, gamma={gamma[i]}, fps={fps[i]:.2f}, lat={lat_ms[i]:.1f}ms"
        plt.annotate(lbl, (x[i], mAP[i]),
                     fontsize=8, xytext=(4,4), textcoords="offset points")

    # baseline と best には追加で注釈
    plt.annotate("Baseline", (x[baseline_idx], mAP[baseline_idx]),
                 fontsize=9, xytext=(10,-10), textcoords="offset points", color="red")
    improve = (mAP[best_idx]-mAP[baseline_idx])/mAP[baseline_idx]*100
    plt.annotate(f"Best (+{improve:.1f}%)", (x[best_idx], mAP[best_idx]),
                 fontsize=9, xytext=(10,10), textcoords="offset points", color="green")

    plt.xlabel(x_label)
    plt.ylabel("mAP ↑ bigger is better")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()

# ====== プロット: Latency版 ======
plot_graph(lat_ms, "Latency (ms) ↓ smaller is better", 
           "mAP vs Latency with Baseline & Best Highlighted")

# ====== プロット: FPS版 ======
plot_graph(fps, "FPS ↑ bigger is better", 
           "mAP vs FPS with Baseline & Best Highlighted")
