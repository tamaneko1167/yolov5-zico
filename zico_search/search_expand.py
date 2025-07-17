import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import yaml
import optuna
from models.yolo import Model
from score import compute_zico_score
import matplotlib
matplotlib.use('Agg')
import plot_utils
import matplotlib.pyplot as plt
import optuna.visualization.matplotlib as vis

# Enable anomaly detection
torch.autograd.set_detect_anomaly(True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def apply_expand_ratio(model_cfg, expand_ratio):
    for layer in model_cfg['backbone'] + model_cfg['head']:
        if layer[2] == 'C3':
            args = layer[3]
            if isinstance(args, list) and len(args) >= 3:
                original_expand = args[2]
                args[2] = round(expand_ratio, 2)
                print(f"Updating C3 layer e={original_expand} to e={args[2]}")
    return model_cfg

def objective(trial):
    expand_ratio = trial.suggest_float('expand_ratio', 0.25, 1.5)

    # Load YAML
    with open('/scratch-local/yolov5-test/models/yolov5n.yaml') as f:
        model_cfg = yaml.safe_load(f)

    # Apply expand_ratio
    model_cfg = apply_expand_ratio(model_cfg, expand_ratio)

    # Build model
    model = Model(model_cfg).to(device)

    # Dummy input
    dummy_input = torch.randn(1, 3, 640, 640).to(device)

    try:
        # Compute ZiCo score
        score = compute_zico_score(model, dummy_input)
        return score
    except Exception as e:
        print(f"[Trial Error] {e}")
        raise optuna.exceptions.TrialPruned()

if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20)

    expands = [t.params["expand_ratio"] for t in study.trials]
    scores = [t.value for t in study.trials]

    plot_utils.plot_param_vs_score(expands, scores, param_name="expand_ratio", filename="expand_vs_zico.png")

    # # Optional visualizations
    # fig1 = vis.plot_param_importances(study)
    # fig1.get_figure().savefig("zico_search/param_importance_expand.png")
    # fig2 = vis.plot_optimization_history(study)
    # fig2.get_figure().savefig("zico_search/optimization_history_expand.png")

    plt.close('all')

    print("Best trial:")
    print(study.best_trial)
