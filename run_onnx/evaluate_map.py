# evaluate_map.py
from mean_average_precision import MetricBuilder
import os
import glob

def evaluate_map(gt_dir, pred_dir):
    metric_fn = MetricBuilder.build_evaluation_metric("map_2d", async_mode=True, num_classes=1)

    for pred_file in glob.glob(os.path.join(pred_dir, "*.txt")):
        base_name = os.path.basename(pred_file)
        gt_file = os.path.join(gt_dir, base_name)

        if not os.path.exists(gt_file):
            continue

        with open(gt_file) as f:
            gt = [list(map(float, line.strip().split())) for line in f]
        with open(pred_file) as f:
            preds = [list(map(float, line.strip().split())) for line in f]

        gt_boxes = [[x - w/2, y - h/2, x + w/2, y + h/2, int(cls)] for cls, x, y, w, h in gt]
        pred_boxes = [[x - w/2, y - h/2, x + w/2, y + h/2, conf, int(cls)] for cls, x, y, w, h, conf in preds]

        metric_fn.add(pred_boxes=pred_boxes, gt_boxes=gt_boxes)

    print(metric_fn.value(iou_thresholds=0.5))

if __name__ == "__main__":
    evaluate_map("../Project/face_data/labels/validation", "pred_txt")
