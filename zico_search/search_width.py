# zico_search/search_width.py
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

#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')  # Force CPU execution to avoid interfering with GPU training


def apply_width_multiple(model_cfg, width_multiple):
    for layer in model_cfg['backbone']:
        if isinstance(layer[3], dict) and 'c2' in layer[3]:
            original_channels = layer[3]['c2']
            new_channels = max(1, int(original_channels * width_multiple))
            print(f"Updating layer c2 from {original_channels} to {new_channels}")
            layer[3]['c2'] = new_channels
    return model_cfg

def objective(trial):
    width_multiple = trial.suggest_float('width_multiple', 0.25, 1.25)

    with open('models/yolov5n.yaml') as f:
        model_cfg = yaml.safe_load(f)

    model_cfg = apply_width_multiple(model_cfg, width_multiple)

    model = Model(model_cfg).to(device)
    dummy_input = torch.randn(1, 3, 640, 640).to(device)

    try:
        score = compute_zico_score_avg(model, dummy_input)
        return score
    except Exception as e:
        print(f"[Trial Error] {e}")
        raise optuna.exceptions.TrialPruned()

if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20)

    widths = [t.params["width_multiple"] for t in study.trials]
    scores = [t.value for t in study.trials]

    plot_utils.plot_param_vs_score(widths, scores, param_name="width_multiple", filename="width_vs_zico.png")
    plot_utils.plot_param_vs_score(widths, scores, param_name="width_multiple", filename="width_vs_zico_spline.png")

    plt.close('all')

    print("Best trial:")
    print(study.best_trial)
