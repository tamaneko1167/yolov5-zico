## Script for practicing, don't use it! 

import os
import torch
from pathlib import Path
import numpy as np
from tqdm import tqdm

from utils.metrics import ap_per_class
from utils.general import xywh2xyxy

# パスの設定
pred_dir = Path("onnx_output/labels")  # 推論結果(txt形式)
gt_dir = Path("../datasets/VOC/labels/test2007")  # Ground Truth (txt形式)

# クラス数とnames（任意）
nc = 20
names = {i: str(i) for i in range(nc)}

# ファイル収集
pred_files = sorted(pred_dir.glob("*.txt"))
gt_files = sorted(gt_dir.glob("*.txt"))

# 検出結果とGTを読み込む
stats = []
for pred_f, gt_f in tqdm(zip(pred_files, gt_files), total=len(gt_files)):
    # Ground truth
    gt = np.loadtxt(gt_f, ndmin=2)
    tcls = gt[:, 0].tolist()  # class
    tbox = xywh2xyxy(gt[:, 1:5])  # xywh to xyxy
    nl = len(gt)

    # Prediction
    pred = np.loadtxt(pred_f, ndmin=2) if pred_f.exists() else np.zeros((0, 6))
    if len(pred):
        #pbox = pred[:, :4]
        pbox = xywh2xyxy(torch.tensor(pred[:, :4]))
        conf = pred[:, 4]
        pcls = pred[:, 5]
    else:
        pbox = conf = pcls = np.array([])

    # IoU-based TP/FP判定
    correct = np.zeros((len(pred), 10), dtype=bool)  # IoU thresholds: 0.5 to 0.95

    image_name = pred_f.stem 
    if nl and len(pred):  # GTもPredもあり
        iouv = torch.linspace(0.5, 0.95, 10)  # IoU thresholds
        correct = np.zeros((len(pred), len(iouv)), dtype=bool)

        from utils.general import box_iou
        ious = box_iou(torch.tensor(tbox), torch.tensor(pbox))  # shape: [num_gt, num_pred]
        ious = ious.cpu()
        
        gt_classes = torch.tensor(tcls).view(-1, 1)  # shape: [num_gt, 1]
        pred_classes = torch.tensor(pcls).view(1, -1)  # shape: [1, num_pred]
        
        correct_class = gt_classes == pred_classes  # shape: [num_gt, num_pred]
        
        for i, iou_thresh in enumerate(iouv):
            x = torch.where((ious >= iou_thresh) & correct_class)  # (gt_idx, pred_idx)ペア
            if len(x[0]) > 0:
                # matches: [gt_idx, pred_idx, iou]
                matches = torch.cat([x[0][:, None], x[1][:, None], ious[x[0], x[1]][:, None]], dim=1).numpy()
                # IoU順にソート
                matches = matches[matches[:, 2].argsort()[::-1]]
                # 各予測について最大IoUだけ残す
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                # 各GTについて最大IoUだけ残す
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
                # 正しい予測として記録
                correct[matches[:, 1].astype(int), i] = True
        
        # per-image metrics (IoU=0.5 only)
        tp_img = correct[:, 0].sum()
        fp_img = len(pred) - tp_img
        fn_img = nl - tp_img
        precision_img = tp_img / (tp_img + fp_img) if (tp_img + fp_img) > 0 else 0
        recall_img = tp_img / (tp_img + fn_img) if (tp_img + fn_img) > 0 else 0

        print(f"[{image_name}] TP: {tp_img}, FP: {fp_img}, FN: {fn_img}, Precision: {precision_img:.3f}, Recall: {recall_img:.3f}, Predictions: {len(pred)}, GTs: {nl}")

    
    elif len(pred) == 0 and nl > 0:
        print(f"{image_name}: No predictions, but GT present. FN={nl}")
    
    elif len(pred) > 0 and nl == 0:
        print(f"{image_name}: Predictions exist but no GT. FP={len(pred)}")

    else:
        print(f"{image_name}: No predictions and no GT.")


    stats.append((correct, conf, pcls, tcls))

# 統合してAP計算
stats = [np.concatenate(x, 0) for x in zip(*stats)]
_, _, _, _, _, ap, ap_class = ap_per_class(*stats, names=names)
map50 = ap.mean() if len(ap) else 0.0

print(f"mAP@0.5: {map50:.4f}")
