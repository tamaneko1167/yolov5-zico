## Inference script for ONNX model with NMS
import os
import time
import onnxruntime as ort
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm

# 設定
onnx_path = "runs/train/yolov5n_voc_baseline/weights/best_with_nms.onnx"
img_dir = "../datasets/VOC/images/test2007"
output_dir = "onnx_output"
txt_output_dir = os.path.join(output_dir, "labels")
img_size = 640
conf_thres = 0.25

os.makedirs(txt_output_dir, exist_ok=True)

# ONNXセッション初期化
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

# 推論対象画像リスト
image_paths = sorted(list(Path(img_dir).glob("*.jpg")))

latencies = []

for img_path in tqdm(image_paths, desc="Running NMS-included ONNX Inference"):
    img, w, h = preprocess(str(img_path))
    start = time.time()
    num_dets, boxes, scores, labels = session.run(None, {input_name: img})
    end = time.time()
    latencies.append((end - start) * 1000)

    num_dets = int(num_dets[0])
    if num_dets == 0:
        continue

    txt_path = os.path.join(txt_output_dir, Path(img_path).stem + ".txt")
    with open(txt_path, "w") as f:
        for i in range(num_dets):
            x1, y1, x2, y2 = boxes[0][i]
            conf = scores[0][i]
            cls_id = labels[0][i]

            if conf < conf_thres:
                continue

            scale_x = w / img_size
            scale_y = h / img_size

            # リサイズされた予測座標 → 元画像サイズにスケーリング
            x1 *= scale_x
            x2 *= scale_x
            y1 *= scale_y
            y2 *= scale_y

            cx = (x1 + x2) / 2 / w
            cy = (y1 + y2) / 2 / h
            bw = (x2 - x1) / w
            bh = (y2 - y1) / h

            # f.write(f"{int(cls_id)} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
            f.write(f"{cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f} {conf:.6f} {int(cls_id)}\n")# => cx cy w h conf cls_id　##include the confidence score

# 平均処理時間表示
print(f"平均FPS: {1000 / np.mean(latencies):.2f}")
print(f"平均Latency: {np.mean(latencies):.2f} ms/frame")
