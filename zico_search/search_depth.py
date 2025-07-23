# zico_search/search_depth.py
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import yaml
import optuna
from models.yolo import Model
from score import compute_zico_score_avg
import matplotlib
matplotlib.use('Agg') 
import plot_utils
import matplotlib.pyplot as plt
import optuna.visualization.matplotlib as vis

# Enable anomaly detection
torch.autograd.set_detect_anomaly(True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def apply_depth_multiple(model_cfg, depth_multiple):
    for layer in model_cfg['backbone']:
        if layer[2] == 'C3':
            original_number = layer[1]
            new_number = max(1, int(original_number * depth_multiple))  # Ensure at least 1
            print(f"Updating C3 layer from n={original_number} to n={new_number}")
            layer[1] = new_number
    return model_cfg

def objective(trial):
    depth_multiple = trial.suggest_float('depth_multiple', 0.33, 1.0)

    # Load YAML
    with open('models/yolov5n.yaml') as f:
        model_cfg = yaml.safe_load(f)

    # Apply depth_multiple
    model_cfg = apply_depth_multiple(model_cfg, depth_multiple)

    # Build model
    model = Model(model_cfg).to(device)

    # Dummy input
    dummy_input = torch.randn(1, 3, 640, 640).to(device)

    try:
        # Compute ZiCo score
        score = compute_zico_score_avg(model, dummy_input, runs=3, seed=42)
        return score
    except Exception as e:
        print(f"[Trial Error] {e}")
        raise optuna.exceptions.TrialPruned()

if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=2000)

    depths = [t.params["depth_multiple"] for t in study.trials]
    scores = [t.value for t in study.trials]

    plot_utils.plot_param_vs_score(depths, scores, param_name="depth_multiple", filename="depth_vs_zico.png")

    plt.close('all') 

    print("Best trial:")
    print(study.best_trial)
