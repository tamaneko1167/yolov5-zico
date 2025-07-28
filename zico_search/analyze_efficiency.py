# zico_search/analyze_efficiency.py
## This script is used to analyze the efficiency of the model by comparing the number of parameters and FLOPs with the ZiCo score.

import os
import sys
import torch
from thop import profile
import yaml
import optuna
from optuna.importance import get_param_importances
import matplotlib.pyplot as plt
import plot_utils

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
    # params = sum(p.numel() for p in model.parameters())

    # with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU],
    #                             record_shapes=True) as prof:
    #     with torch.profiler.record_function("model_inference"):
    #         model(dummy_input)
    # flops = sum([e.cpu_time_total for e in prof.key_averages() if e.cpu_time_total is not None])
    flops, params = profile(model, inputs=(dummy_input,), verbose=False)

    return zico, params, flops

def objective(trial):
    depth = trial.suggest_float("depth_multiple", 0.33, 1.0)
    width = trial.suggest_float("width_multiple", 0.25, 1.0)

    with open("models/yolov5n.yaml") as f:
        cfg = yaml.safe_load(f)
    cfg = apply_cfg_changes(cfg, depth, width)

    try:
        model = Model(cfg).to(device)
        zico, params, flops = compute_model_metrics(model)
        results.append((zico, params, flops,depth, width))  
        print(f"[Trial] depth={depth:.2f}, width={width:.2f} => ZiCo={zico:.2f}, Params={params/1e6:.2f}M, FLOPs={flops/1e6:.2f}M")
        return params, flops, zico
    except Exception as e:
        print(f"[Trial Error] {e}")
        raise optuna.exceptions.TrialPruned()

from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def plot_depth_flops_params(results_with_hyperparams, save_path):
    depth_list = [r[3] for r in results_with_hyperparams]
    flops_list = [r[2] for r in results_with_hyperparams]
    params_list = [r[1] for r in results_with_hyperparams]
    zico_scores = [r[0] for r in results_with_hyperparams]

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    scatter = ax.scatter(flops_list, params_list, zico_scores, c=depth_list, cmap='plasma', s=40)
    fig.colorbar(scatter, label="ZiCo Score")

    ax.set_xlabel("FLOPs")
    ax.set_ylabel("Params")
    ax.set_zlabel("ZiCo Score")
    plt.title("FLOPs × Params (Depth Score color)")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_width_flops_params(results_with_hyperparams, save_path):

    width_list = [r[4] for r in results_with_hyperparams]
    flops_list = [r[2] for r in results_with_hyperparams]
    params_list = [r[1] for r in results_with_hyperparams]
    zico_scores = [r[0] for r in results_with_hyperparams]

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    scatter = ax.scatter(flops_list, params_list, zico_scores, c=width_list, cmap='plasma', s=40)
    fig.colorbar(scatter, label="ZiCo Score")

    ax.set_xlabel("FLOPs")
    ax.set_ylabel("Params")
    ax.set_zlabel("ZiCo Score")
    plt.title("FLOPs × Params (Width Score color)")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()




if __name__ == "__main__":
    study = optuna.create_study(directions=["maximize", "minimize", "minimize"])
    study.optimize(objective, n_trials=2000)  # 試行回数をここで調整

    # ZiCo vs FLOPs
    optuna.visualization.plot_pareto_front(study, target_names=["Params", "FLOPs", "ZiCo"]).write_image(os.path.join(PLOT_DIR, "pareto_zico_flops_params.png"))
    # multi-objective の場合は target を明示
    imp_zico   = get_param_importances(study, target=lambda t: t.values[2])   # 例: values=[params, flops, zico]
    imp_params = get_param_importances(study, target=lambda t: t.values[0])
    imp_flops  = get_param_importances(study, target=lambda t: t.values[1])

    print("ZiCo importance:", imp_zico)
    print("Params importance:", imp_params)
    print("FLOPs importance:", imp_flops)

    plot_depth_flops_params(results,save_path=os.path.join(PLOT_DIR, "depth_flops_params.png"))
    plot_width_flops_params(results,save_path=os.path.join(PLOT_DIR, "width_flops_params.png"))
    # Width vs 各指標の可視化
    plot_utils.plot_width_vs_metric(
        results, metric_index=0, ylabel="ZiCo Score",
        save_path=os.path.join(PLOT_DIR, "width_vs_zico.png")
    )

    plot_utils.plot_width_vs_metric(
        results, metric_index=1, ylabel="Params",
        save_path=os.path.join(PLOT_DIR, "width_vs_params.png")
    )

    plot_utils.plot_width_vs_metric(
        results, metric_index=2, ylabel="FLOPs",
        save_path=os.path.join(PLOT_DIR, "width_vs_flops.png")
    )
