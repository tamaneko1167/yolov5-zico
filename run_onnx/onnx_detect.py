import sys
import os
import glob
import json
import cv2
import numpy as np
import torch
import onnxruntime
import torchvision
from tqdm import tqdm
import time

names = ['face'] 

def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), stride=32):
    shape = im.shape[:2]  # [h, w]
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    r = min(r, 1.0)
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw /= 2
    dh /= 2
    im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im, (r, r), (dw, dh)

def xywh2xyxy(x):
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y

def scale_coords(img1_shape, coords, img0_shape):
    gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
    pad = ((img1_shape[1] - img0_shape[1] * gain) / 2,
           (img1_shape[0] - img0_shape[0] * gain) / 2)
    coords[..., [0, 2]] -= pad[0]
    coords[..., [1, 3]] -= pad[1]
    coords[..., :4] /= gain
    coords[..., [0, 2]] = np.clip(coords[..., [0, 2]], 0, img0_shape[1])
    coords[..., [1, 3]] = np.clip(coords[..., [1, 3]], 0, img0_shape[0])
    return coords

def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45):
    bs = prediction.shape[0]
    nc = prediction.shape[2] - 5
    xc = prediction[..., 4] > conf_thres
    output = [torch.zeros((0, 6))] * bs
    for xi, x in enumerate(prediction):
        x = x[xc[xi]]
        if not x.shape[0]:
            continue
        x[:, 5:] *= x[:, 4:5]
        box = xywh2xyxy(x[:, :4])
        conf, j = x[:, 5:].max(1, keepdim=True)
        x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]
        if not x.shape[0]:
            continue
        i = torchvision.ops.nms(x[:, :4], x[:, 4], iou_thres)
        output[xi] = x[i]
    return output

def onnx_detect(onnx_path, img_dir, output_dir, vis_dir):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True) 
    session = onnxruntime.InferenceSession(onnx_path)
    img_paths = glob.glob(os.path.join(img_dir, "**/*.jpg"), recursive=True)

    total_time = 0
    all_times = []

    for img_path in tqdm(img_paths, desc="Running inference"):
        img0 = cv2.imread(img_path)
        img, (r_w, r_h), (dw, dh) = letterbox(img0)
        img_tensor = img.transpose(2, 0, 1)[::-1]
        img_tensor = np.ascontiguousarray(img_tensor, dtype=np.float32) / 255.0
        img_tensor = np.expand_dims(img_tensor, axis=0)
        input_shape = img_tensor.shape[2:]

        # 推論時間測定開始
        t1 = time.time()
        pred = session.run(None, {session.get_inputs()[0].name: img_tensor})[0]
        t2 = time.time()
        inference_time = t2 - t1
        all_times.append(inference_time)
        total_time += inference_time

        pred = session.run(None, {session.get_inputs()[0].name: img_tensor})[0]
        pred = torch.from_numpy(pred)
        det = non_max_suppression(pred)[0]

        results = []
        if len(det):
            det[:, :4] = torch.tensor(scale_coords(input_shape, det[:, :4].numpy(), img0.shape)).round()
            for *xyxy, conf, cls in det:
                results.append({
                    "box": [float(x.item()) for x in xyxy],
                    "score": float(conf.item()),
                    "class_id": int(cls.item())
                })

                label = f"{cls}:{conf:.2f}"
                cv2.rectangle(img0,(int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])),(0, 255, 0),2)
                cv2.putText(img0, label, (int(xyxy[0]), int(xyxy[1]) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        base = os.path.splitext(os.path.basename(img_path))[0]
        out_path = os.path.join(output_dir, f"{base}.json")
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)

        vis_path = os.path.join(vis_dir, base + "_vis.jpg")
        cv2.imwrite(vis_path, img0)
        
    print(f"\n 推論時間: 合計 {total_time:.2f} 秒, 平均 {total_time / len(img_paths):.4f} 秒/画像")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx_path", type=str, required=True)
    parser.add_argument("--img_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()
    onnx_detect(
        onnx_path=args.onnx_path,
        img_dir=args.img_dir,
        output_dir=args.output_dir,
        vis_dir="vis_img"
    )