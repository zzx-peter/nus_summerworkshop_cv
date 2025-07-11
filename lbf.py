import cv2
import os
import numpy as np
from collections import defaultdict

# 加载人脸检测器和关键点模型
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
facemark = cv2.face.createFacemarkLBF()
facemark.loadModel('lbfmodel.yaml')

# 统计信息
total_count = defaultdict(int)
success_count = defaultdict(int)

def process_image_to_keypoints_canvas(image_path, output_path, class_name):
    image = cv2.imread(image_path)
    if image is None:
        print(f"加载失败：{image_path}")
        return

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    total_count[class_name] += 1  # 每张图片都记一次

    if len(faces) == 0:
        print(f"未检测到人脸：{image_path}")
        return  # 跳过保存

    _, landmarks = facemark.fit(gray, faces)
    canvas = np.ones_like(image) * 255

    for (x, y) in landmarks[0][0]:
        cv2.circle(canvas, (int(x), int(y)), 1, (0, 0, 0), -1)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, canvas)

    success_count[class_name] += 1  # 成功保存

def batch_convert_dataset(input_root, output_root):
    valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    for root, _, files in os.walk(input_root):
        for fname in files:
            if not any(fname.lower().endswith(ext) for ext in valid_extensions):
                continue
            input_path = os.path.join(root, fname)
            rel_path = os.path.relpath(input_path, input_root)
            output_path = os.path.join(output_root, rel_path)

            # 提取 class name
            class_name = rel_path.split(os.sep)[0]
            process_image_to_keypoints_canvas(input_path, output_path, class_name)

if __name__ == "__main__":
    input_dir = "facial_expression_dataset/test"
    output_dir = "facial_expression_dataset_lbf/test"
    batch_convert_dataset(input_dir, output_dir)

    # 输出统计结果
    print("\n关键点检测统计结果:")
    for cls in sorted(total_count.keys()):
        total = total_count[cls]
        success = success_count[cls]
        print(f"类 [{cls}]: 总数 = {total}，成功检测 = {success}，成功率 = {success / total:.2%}")
