## This script is used to visualize the valudation labels on images for testing .
import os
import cv2
import glob

def lavel_test(img_dir, label_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    img_paths = glob.glob(os.path.join(img_dir, "*.jpg"))  # 画像一覧を取得
    for img_path in img_paths:
        base = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(label_dir, base + ".txt")
        output_path = os.path.join(output_dir, base + "_label.jpg")

        if not os.path.exists(label_path):
            print(f"Label not found for: {img_path}")
            continue

        img = cv2.imread(img_path)
        if img is None:
            print(f"Could not read: {img_path}")
            continue

        h, w, _ = img.shape
        with open(label_path, 'r') as f:
            lines = f.readlines()

        for line in lines:
            class_id, x_center, y_center, box_width, box_height = map(float, line.strip().split())
            x1 = int((x_center - box_width / 2) * w)
            y1 = int((y_center - box_height / 2) * h)
            x2 = int((x_center + box_width / 2) * w)
            y2 = int((y_center + box_height / 2) * h)

            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, f'class {int(class_id)}', (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

        cv2.imwrite(output_path, img)
        print(f"Saved: {output_path}")


if __name__ == "__main__":
    lavel_test("../../project/face_data/images/validation", "../../project/face_data/labels/validation", "label_test")