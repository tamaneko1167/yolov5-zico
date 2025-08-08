# zico_search/analyze_efficiency.py
## Analyze the effect of depth and width on model efficiency using ZiCo score
## This finally suggest that width has no significant effect on efficiency

import os, sys, torch, yaml, optuna, numpy as np
import matplotlib.pyplot as plt
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.yolo import Model
from score import compute_zico_score_avg

device = torch.device('cpu')
PLOT_DIR = os.path.join(os.path.dirname(__file__), "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

# ===== 探索グリッド（相互作用を見る用に離散化） =====
DEPTH_CAND = [0.5, 0.7, 0.9, 1.0, 1.1]
WIDTH_CAND  = [0.25,0.33,0.5, 0.7, 0.85, 1.0]

results = []  # (zico, params, latency_us, depth, width)

def apply_cfg_changes(cfg, depth_multiple, width_multiple):
    # 深さ：C3の繰り返し回数 n をスケール
    for layer in cfg['backbone']:
        if layer[2] == 'C3':
            layer[1] = max(1, int(round(layer[1] * depth_multiple)))

    # 幅：本コードでは C3 内の c2 をスケールしている前提（あなたのcfg仕様に合わせる）
    for layer in cfg['backbone']:
        if isinstance(layer[3], dict) and 'c2' in layer[3]:
            layer[3]['c2'] = max(1, int(round(layer[3]['c2'] * width_multiple)))
        # もし 'c1' など他のキーもある場合はここで同様に調整
    return cfg

def compute_model_metrics(model, dummy_input):
    # ---- ZiCo ----
    zico = compute_zico_score_avg(model, dummy_input, runs=3, seed=42)

    # ---- Params ----
    params = sum(p.numel() for p in model.parameters())

    # ---- Latency proxy (CPU time) ※FLOPsではない ----
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU], record_shapes=False) as prof:
        with torch.profiler.record_function("model_inference"):
            _ = model(dummy_input)
    latency_us = sum(e.cpu_time_total for e in prof.key_averages() if e.cpu_time_total is not None)

    # ---- FLOPs (任意: thopが入っていれば使う) ----
    flops = None
    try:
        from thop import profile
        model.eval()
        with torch.no_grad():
            macs, _ = profile(model, inputs=(dummy_input,), verbose=False)
        flops = 2 * macs  # MACs→FLOPs換算
    except Exception:
        pass

    return zico, params, latency_us, flops

def objective(trial):
    depth = trial.suggest_categorical("depth_multiple", DEPTH_CAND)
    width = trial.suggest_categorical("width_multiple", WIDTH_CAND)

    with open("models/yolov5n.yaml") as f:
        cfg = yaml.safe_load(f)
    cfg = apply_cfg_changes(cfg, depth, width)

    try:
        model = Model(cfg).to(device)
        dummy_input = torch.randn(1, 3, 640, 640, device=device)

        zico, params, latency_us, flops = compute_model_metrics(model, dummy_input)
        results.append((zico, params, latency_us, depth, width, flops))
        print(f"[Trial] d={depth:.2f}, w={width:.2f} => ZiCo={zico:.2f}, "
              f"Params={params/1e6:.2f}M, Latency(us)={latency_us:.0f}, FLOPs={flops/1e9 if flops else None}G")

        # まずは単純にZiCo最大化（後で重み付けも可）
        return zico
    except Exception as e:
        print(f"Failed trial: {e}")
        return float('-inf')

def plot_heatmaps(results):
    # グリッド整形
    depths = sorted(set(r[3] for r in results))
    widths = sorted(set(r[4] for r in results))
    D, W = len(depths), len(widths)

    zico_grid = np.zeros((D, W))
    param_grid = np.zeros((D, W))
    lat_grid = np.zeros((D, W))
    has_flops = any(r[5] is not None for r in results)
    flops_grid = np.zeros((D, W)) if has_flops else None

    idx_d = {d:i for i,d in enumerate(depths)}
    idx_w = {w:i for i,w in enumerate(widths)}

    for zico, params, lat, d, w, flops in results:
        i, j = idx_d[d], idx_w[w]
        zico_grid[i, j] = zico
        param_grid[i, j] = params / 1e6
        lat_grid[i, j] = lat / 1e6  # ms相当（us→秒: /1e6）
        if has_flops and flops is not None:
            flops_grid[i, j] = flops / 1e9  # GFLOPs

    # ZiCoヒートマップ + 等高線
    plt.figure()
    im = plt.imshow(zico_grid, origin='lower', cmap='viridis',
                    extent=[0, W, 0, D], aspect='auto')
    plt.colorbar(im, label='ZiCo')
    # 等高線（パラメータ）
    CS = plt.contour(np.arange(W), np.arange(D), param_grid, colors='white', linewidths=0.5)
    plt.clabel(CS, inline=True, fontsize=7, fmt='%.1fM')
    plt.xticks(np.arange(W), [f"{w:.2f}" for w in widths], rotation=0)
    plt.yticks(np.arange(D), [f"{d:.2f}" for d in depths])
    plt.xlabel("width_multiple")
    plt.ylabel("depth_multiple")
    plt.title("ZiCo Heatmap (white contours: Params M)")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "zico_heatmap_params.png"), dpi=300)

    # Latency 等高線版
    plt.figure()
    im = plt.imshow(zico_grid, origin='lower', cmap='viridis',
                    extent=[0, W, 0, D], aspect='auto')
    plt.colorbar(im, label='ZiCo')
    CS = plt.contour(np.arange(W), np.arange(D), lat_grid, colors='white', linewidths=0.5)
    plt.clabel(CS, inline=True, fontsize=7, fmt='%.2f s')
    plt.xticks(np.arange(W), [f"{w:.2f}" for w in widths])
    plt.yticks(np.arange(D), [f"{d:.2f}" for d in depths])
    plt.xlabel("width_multiple"); plt.ylabel("depth_multiple")
    plt.title("ZiCo Heatmap (white contours: Latency ~s)")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "zico_heatmap_latency.png"), dpi=300)

    # FLOPs あるなら別図も
    if has_flops:
        plt.figure()
        im = plt.imshow(zico_grid, origin='lower', cmap='viridis',
                        extent=[0, W, 0, D], aspect='auto')
        plt.colorbar(im, label='ZiCo')
        CS = plt.contour(np.arange(W), np.arange(D), flops_grid, colors='white', linewidths=0.5)
        plt.clabel(CS, inline=True, fontsize=7, fmt='%.1fG')
        plt.xticks(np.arange(W), [f"{w:.2f}" for w in widths])
        plt.yticks(np.arange(D), [f"{d:.2f}" for d in depths])
        plt.xlabel("width_multiple"); plt.ylabel("depth_multiple")
        plt.title("ZiCo Heatmap (white contours: GFLOPs)")
        plt.tight_layout()
        plt.savefig(os.path.join(PLOT_DIR, "zico_heatmap_flops.png"), dpi=300)

if __name__ == "__main__":
    # グリッド探索っぽく回したいのでcategorical + 多試行
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials= len(DEPTH_CAND) * len(WIDTH_CAND) * 3)  # 同一点を数回測ってノイズ平均化

    print("\nBest trial:")
    print(study.best_trial)

    plot_heatmaps(results)
