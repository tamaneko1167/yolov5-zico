# zico_search/analyze_efficiency.py
## This script is used to analyze the efficiency of the model by comparing the number of parameters and FLOPs with the ZiCo score.

import os
import sys
import torch
import yaml
import optuna
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.yolo import Model
from score import compute_zico_score_avg

device = torch.device('cpu')
PLOT_DIR = os.path.join(os.path.dirname(__file__), "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

results = []  # For plotting later

def apply_cfg_changes(cfg, depth_multiple, width_multiple):
    for layer in cfg['backbone']:
        if layer[2] == 'C3':
            layer[1] = max(1, int(layer[1] * depth_multiple))
    for layer in cfg['backbone']:
        if isinstance(layer[3], dict) and 'c2' in layer[3]:
            layer[3]['c2'] = max(1, int(layer[3]['c2'] * width_multiple))
    return cfg

def compute_model_metrics(model):
    dummy_input = torch.randn(1, 3, 640, 640).to(device)

    zico = compute_zico_score_avg(model, dummy_input, runs=3, seed=42)
    params = sum(p.numel() for p in model.parameters())

    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU],
                                record_shapes=True) as prof:
        with torch.profiler.record_function("model_inference"):
            model(dummy_input)
    flops = sum([e.cpu_time_total for e in prof.key_averages() if e.cpu_time_total is not None])

    return zico, params, flops

def objective(trial):
    depth = trial.suggest_float("depth_multiple", 0.33, 1.0)
    width = 0.5  # 固定

    with open("models/yolov5n.yaml") as f:
        cfg = yaml.safe_load(f)
    cfg = apply_cfg_changes(cfg, depth, width)

    try:
        model = Model(cfg).to(device)
        zico, params, flops = compute_model_metrics(model)
        results.append((zico, params, flops, depth))  # depth を記録
        print(f"[Trial] depth={depth:.2f} => ZiCo={zico:.2f}, Params={params/1e6:.2f}M, FLOPs={flops/1e6:.2f}M")
        return zico
    except Exception as e:
        print(f"Failed trial: {e}")
        return float('-inf')

def plot_results(results):
    zico_scores = [r[0] for r in results]
    params_list = [r[1] / 1e6 for r in results]  # in Millions
    flops_list = [r[2] / 1e6 for r in results]   # in Millions
    depth_values = [r[3] for r in results]       # for color mapping

    # FLOPs vs ZiCo（色分け）
    plt.figure()
    sc = plt.scatter(flops_list, zico_scores,s=5, c=depth_values, cmap='viridis', alpha=0.8)
    plt.xlabel("FLOPs (Millions)")
    plt.ylabel("ZiCo Score")
    plt.title("ZiCo Score vs FLOPs (colored by depth)")
    plt.colorbar(sc, label='depth_multiple')
    plt.grid(True)
    plt.savefig(os.path.join(PLOT_DIR, "zico_vs_flops_by_depth.png"), dpi=300)

    # Params vs ZiCo（色分け）
    plt.figure()
    sc = plt.scatter(params_list, zico_scores, s=5, c=depth_values, cmap='viridis', alpha=0.8)
    plt.xlabel("Number of Parameters (Millions)")
    plt.ylabel("ZiCo Score")
    plt.title("ZiCo Score vs Params (colored by depth)")
    plt.colorbar(sc, label='depth_multiple')
    plt.grid(True)
    plt.savefig(os.path.join(PLOT_DIR, "zico_vs_params_by_depth.png"), dpi=300)

if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=2000)  # 試行回数をここで調整

    print("\nBest trial:")
    print(study.best_trial)

    plot_results(results)
