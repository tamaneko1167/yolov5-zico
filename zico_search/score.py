# zico_search/score.py

import torch
import torch.nn as nn
import numpy as np
from thop import profile

def compute_zico_score(model, input_tensor):
    model.train()  # 必ずtrainモードに。eval()はinplace batchnormやdropoutでgrad問題の原因になる
    input_tensor.requires_grad_()  # 明示的にrequires_grad=Trueにする

    # 可能な限り in-place 操作の影響を受けにくくする
    output = model(input_tensor)[0]  # prediction head (batch, anchors, grid, grid, num_outputs)

    loss = output.sum()
    loss.backward()

    scores = []
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad = param.grad.view(-1).detach().cpu().numpy()
            if grad.size > 0:
                scores.append(np.std(grad))
    
    if len(scores) == 0:
        raise ValueError("No gradients found in model parameters.")

    return float(np.mean(scores))  # ZiCoスコア（分散の平均）

def get_flops_and_params(model, input_tensor):
    macs, params = profile(model, inputs=(input_tensor,), verbose=False)
    return macs / 1e6, params / 1e6  # MFLOPs, MParams