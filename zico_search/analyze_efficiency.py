# zico_search/analyze_efficiency.py
## This script is used to analyze the efficiency of the model by comparing the number of parameters and FLOPs with the ZiCo score.

import os
import sys
import torch
from thop import profile
import yaml
import numpy as np
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
    flops, params = profile(model, inputs=(dummy_input,), verbose=False)

    return zico, params, flops

def objective(trial):
    # depth = trial.suggest_float("depth_multiple", 0.33, 1.0) # Depth is discreted because it's rounded to nearest integer
    depth_list = np.round(np.arange(0.3, 1.05, 0.05), 2).tolist()
    depth = trial.suggest_categorical("depth_multiple", depth_list)
    width = 0.5  # Fixed width because we figured out that width has no impact on the score

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

if __name__ == "__main__":
    study = optuna.create_study(directions=["maximize", "minimize", "minimize"])
    study.optimize(objective, n_trials=2000)  # 試行回数をここで調整

    # # ZiCo vs FLOPs
    # optuna.visualization.plot_pareto_front(study, target_names=["Params", "FLOPs", "ZiCo"]).write_image(os.path.join(PLOT_DIR, "pareto_zico_flops_params.png"))
    # # multi-objective の場合は target を明示
    # imp_zico   = get_param_importances(study, target=lambda t: t.values[2])   # 例: values=[params, flops, zico]
    # imp_params = get_param_importances(study, target=lambda t: t.values[0])
    # imp_flops  = get_param_importances(study, target=lambda t: t.values[1])

    # print("ZiCo importance:", imp_zico)
    # print("Params importance:", imp_params)
    # print("FLOPs importance:", imp_flops)

    plot_utils.plot_depth_flops_params(results,save_path=os.path.join(PLOT_DIR, "flops_params_zico_colored_by_depth.png"))

    # Depth Multiple vs each metric
    plot_utils.plot_param_vs_metric(results, 3, 0, "Depth Multiple", "ZiCo Score", os.path.join(PLOT_DIR, "depth_vs_zico.png"))

    # Depth Multiple vs Params
    plot_utils.plot_param_vs_metric(results, 3, 1, "Depth Multiple", "Params", os.path.join(PLOT_DIR, "depth_vs_params.png"))

    # Depth Multiple vs FLOPs
    plot_utils.plot_param_vs_metric(results, 3, 2, "Depth Multiple", "FLOPs", os.path.join(PLOT_DIR, "depth_vs_flops.png"))

