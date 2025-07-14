import cv2
import numpy as np
import torch
import onnxruntime
import math
import time
import json
import torchvision

names = ['person', 'car', 'dog']  # ←学習時のクラス名に合わせてください

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

def scale_boxes(img1_shape, boxes, img0_shape):
    gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
    pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2
    boxes[..., [0, 2]] -= pad[0]
    boxes[..., [1, 3]] -= pad[1]
    boxes[..., :4] /= gain
    boxes[..., [0, 2]] = np.clip(boxes[..., [0, 2]], 0, img0_shape[1])
    boxes[..., [1, 3]] = np.clip(boxes[..., [1, 3]], 0, img0_shape[0])
    return boxes

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

def run_onnx_inference(onnx_path, image_path, result_txt_path):
    session = onnxruntime.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    img0 = cv2.imread(image_path)
    img, _, _ = letterbox(img0, new_shape=(640, 640))
    img = img.transpose(2, 0, 1)[::-1]
    img = np.ascontiguousarray(img, dtype=np.float32) / 255.0
    img = np.expand_dims(img, axis=0)

    t0 = time.time()
    y = session.run(None, {session.get_inputs()[0].name: img})
    t1 = time.time()
    print(f"Inference time: {t1 - t0:.4f} seconds")

    pred = torch.from_numpy(y[0])
    pred = non_max_suppression(pred, 0.25, 0.45)

    result_list = []
    for det in pred:
        if len(det):
            det[:, :4] = torch.tensor(scale_boxes(img.shape[2:], det[:, :4].numpy(), img0.shape)).round()
            for *xyxy, conf, cls in det:
                box = [float(x.item()) for x in xyxy]
                result_list.append({
                    "box": box,
                    "score": float(conf.item()),
                    "class_id": int(cls.item()),
                    "class_name": names[int(cls.item())] if int(cls.item()) < len(names) else "unknown"
                })

    with open(result_txt_path, "w") as f:
        json.dump(result_list, f, indent=2)

    print(f"Results saved to {result_txt_path}")

if __name__ == '__main__':
    run_onnx_inference("runs/train/exp2/weights/best.onnx", "data/images/bus.jpg", "result.json")
