## Evaluate mAP for ONNX model predictions
# Use after inference with "run_onnx_with_nms.py"

import os
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from utils.metrics import ap_per_class
from val import process_batch
from utils.general import xywh2xyxy

# 設定
pred_dir = Path("onnx_output/labels")  # ONNX 推論結果（txtファイル）
gt_dir = Path("../datasets/VOC/labels/test2007")  # GT ラベル
nc = 20  # クラス数
iouv = torch.linspace(0.5, 0.95, 10)  # IoU thresholds
niou = iouv.numel()

def load_labels(file_path):
    if not file_path.exists():
        return torch.zeros((0, 6))
    data = np.loadtxt(file_path, ndmin=2)
    return torch.tensor(data, dtype=torch.float32)

stats = []

for pred_file in tqdm(list(pred_dir.glob("*.txt")), desc="Evaluating mAP"):
    image_id = pred_file.stem
    gt_file = gt_dir / f"{image_id}.txt"
    if not gt_file.exists():
        continue

    preds = load_labels(pred_file)
    gts = load_labels(gt_file)

    if preds.shape[0] == 0:
        preds = preds.reshape(0, 6)
    if gts.shape[0] == 0:
        continue

    # pred: [x_center, y_center, w, h, conf, cls] → [x1, y1, x2, y2, conf, cls]
    preds[:, :4] = xywh2xyxy(preds[:, :4])
    preds = preds[:, [0, 1, 2, 3, 4, 5]]  # [x1, y1, x2, y2, conf, cls]

    # gt: [cls, x_center, y_center, w, h] → [cls, x1, y1, x2, y2]
    gts[:, 1:5] = xywh2xyxy(gts[:, 1:5])

    correct = process_batch(preds, gts, iouv)
    stats.append((correct, preds[:, 4], preds[:, 5], gts[:, 0]))

# mAP計算
stats = [torch.cat(x, 0).cpu().numpy() for x in zip(*stats)]
if len(stats) and stats[0].any():
    # 修正済み：names は空でもエラーを出さないようにする
    def safe_ap_per_class(tp, conf, pred_cls, target_cls, **kwargs):
        try:
            return ap_per_class(tp, conf, pred_cls, target_cls, **kwargs)
        except Exception:
            return 0, 0, 0, 0, 0, 0, []

    tp, fp, p, r, f1, ap, ap_class = safe_ap_per_class(*stats, plot=False, save_dir='.', names={})
    map50, map = ap[:, 0].mean(), ap.mean()
else:
    p = r = ap = f1 = ap_class = map50 = map = 0

# 結果出力
print(f"\nmAP@0.5: {map50:.4f}")
print(f"mAP@0.5:0.95: {map:.4f}")
