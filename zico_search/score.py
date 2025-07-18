# zico_search/score.py

import torch
import torch.nn as nn
import numpy as np
from thop import profile

import random

def compute_zico_score_avg(model, input_tensor, runs=3, seed=42):
    scores = []
    for i in range(runs):
        # 乱数を毎回固定（PyTorch, NumPy, Python組み込み）
        torch.manual_seed(seed + i)
        np.random.seed(seed + i)
        random.seed(seed + i)

        # modelを毎回初期化し直す方が厳密だけど、今回はそのまま使う
        model.zero_grad()
        input_tensor.requires_grad_()

        model.train()
        output = model(input_tensor)[0]
        loss = output.sum()
        loss.backward()

        score_list = []
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad = param.grad.view(-1).detach().cpu().numpy()
                if grad.size > 0:
                    score_list.append(np.std(grad))
        if len(score_list) == 0:
            raise ValueError("No gradients found in model parameters.")

        scores.append(np.mean(score_list))

    return float(np.mean(scores))


def get_flops_and_params(model, input_tensor):
    macs, params = profile(model, inputs=(input_tensor,), verbose=False)
    return macs / 1e6, params / 1e6  # MFLOPs, MParams