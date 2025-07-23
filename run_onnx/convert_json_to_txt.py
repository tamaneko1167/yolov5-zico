import os
import json
import cv2

def json_to_yolo_txt(pred_json_dir, output_txt_dir, image_dir):
    os.makedirs(output_txt_dir, exist_ok=True)

    for json_file in os.listdir(pred_json_dir):
        if not json_file.endswith(".json"):
            continue

        # 対応する画像を読み込んでサイズ取得
        base_name = os.path.splitext(json_file)[0]
        image_path = os.path.join(image_dir, base_name + ".jpg")
        img = cv2.imread(image_path)
        if img is None:
            print(f"Image not found or unreadable: {image_path}")
            continue
        img_h, img_w = img.shape[:2]

        with open(os.path.join(pred_json_dir, json_file)) as f:
            preds = json.load(f)

        if len(preds) == 0:
            continue

        txt_lines = []
        for pred in preds:
            class_id = pred["class_id"]
            x1, y1, x2, y2 = pred["box"]
            x_center = ((x1 + x2) / 2) / img_w
            y_center = ((y1 + y2) / 2) / img_h
            width = (x2 - x1) / img_w
            height = (y2 - y1) / img_h
            txt_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

        with open(os.path.join(output_txt_dir, base_name + ".txt"), "w") as f:
            f.write("\n".join(txt_lines))

# 実行
if __name__ == "__main__":
    json_to_yolo_txt("pred_json", "pred_txt", "../../project/face_data/images/validation")
