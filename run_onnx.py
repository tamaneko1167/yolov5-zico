import os
import time
import onnxruntime as ort
from onnxruntime_extensions import ops
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm

# 設定
onnx_path = "runs/train/yolov5n_voc_baseline/weights/best.onnx"
img_dir = "../datasets/VOC/images/test2007"
#img_dir = "yolov5/data/images"
output_dir = "onnx_output"
txt_output_dir = os.path.join(output_dir, "labels")
img_size = 640
conf_thres = 0.25
iou_thres = 0.6

os.makedirs(txt_output_dir, exist_ok=True)

# ONNX推論用セッション
session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name

# 前処理関数
def preprocess(img_path):
    img0 = cv2.imread(img_path)
    img = cv2.resize(img0, (img_size, img_size))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.transpose(2, 0, 1).astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)
    return img, img0.shape[1], img0.shape[0]

# 非最大抑制（簡易版）
def nms(pred, conf_thres=0.25, iou_thres=0.45):
    boxes = pred[pred[:, 4] > conf_thres]
    if len(boxes) == 0:
        return []
    boxes = boxes[np.argsort(-boxes[:, 4])]
    keep = []
    while boxes.shape[0]:
        box = boxes[0]
        keep.append(box)
        if boxes.shape[0] == 1:
            break
        ious = iou(box, boxes[1:])
        boxes = boxes[1:][ious < iou_thres]
    return keep

def nms_with_extensions(boxes, scores, iou_threshold=0.6, score_threshold=0.25, max_output=100):
    # onnxruntime-extensions の NonMaxSuppression
    nms = ops.non_max_suppression(
        boxes[np.newaxis, :, :].astype(np.float32),    # [1, num_boxes, 4]
        scores[np.newaxis, np.newaxis, :].astype(np.float32),  # [1, 1, num_boxes]
        max_output_boxes_per_class=max_output,
        iou_threshold=iou_threshold,
        score_threshold=score_threshold
    )
    return nms.numpy()

def classwise_nms(boxes, iou_thres=0.45):
    if boxes.shape[0] == 0:
        return []
    result = []
    class_ids = np.unique(boxes[:, 5])
    for cls_id in class_ids:
        cls_boxes = boxes[boxes[:, 5] == cls_id]
        cls_boxes = cls_boxes[np.argsort(-cls_boxes[:, 4])]
        keep = nms(cls_boxes, iou_thres=iou_thres)
        result.extend(keep)
    return result

def iou(box, boxes):
    # [x1, y1, x2, y2]
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    area1 = (box[2] - box[0]) * (box[3] - box[1])
    area2 = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return inter / (area1 + area2 - inter + 1e-6)

# 画像取得
image_paths = sorted(list(Path(img_dir).glob("*.jpg")))

latencies = []

for img_path in tqdm(image_paths, desc="Running ONNX Inference"):
    img, w, h = preprocess(str(img_path))
    start = time.time()
    pred = session.run(None, {input_name: img})[0]
    end = time.time()

    latency = (end - start) * 1000
    latencies.append(latency)

    pred = np.squeeze(pred)
    if len(pred.shape) != 2:
        continue
    
    filtered = []
    for det in pred:
        x1, y1, x2, y2 = det[:4]
        obj_conf = det[4]
        class_probs = det[5:]

        cls_id = np.argmax(class_probs)
        cls_conf = class_probs[cls_id]
        conf = obj_conf * cls_conf

        if conf > conf_thres:
            filtered.append([x1, y1, x2, y2, conf, cls_id])

    filtered = np.array(filtered)

    txt_path = os.path.join(txt_output_dir, Path(img_path).stem + ".txt")
    with open(txt_path, "w") as f:
        if filtered.shape[0] > 0:
            boxes = classwise_nms(filtered, iou_thres)
            for det in boxes:
                x1, y1, x2, y2 = det[0:4]
                conf = det[4]
                cls_id = det[5]

                cx = (x1 + x2) / 2 / w
                cy = (y1 + y2) / 2 / h
                bw = (x2 - x1) / w
                bh = (y2 - y1) / h

                f.write(f"{int(cls_id)} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

# 結果表示
print(f"平均FPS: {1000 / np.mean(latencies):.2f}")
print(f"平均Latency: {np.mean(latencies):.2f} ms/frame")