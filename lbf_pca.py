import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.decomposition import PCA

# 加载人脸检测器和关键点模型
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
facemark = cv2.face.createFacemarkLBF()
facemark.loadModel('lbfmodel.yaml')

# 统计信息
total_count = defaultdict(int)
success_count = defaultdict(int)

# 保存提取到的关键点向量
all_keypoints = []
all_image_paths = []

def process_image_to_keypoints_canvas(image_path, output_path, class_name):
    image = cv2.imread(image_path)
    if image is None:
        print(f"加载失败：{image_path}")
        return

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    total_count[class_name] += 1

    if len(faces) == 0:
        print(f"未检测到人脸：{image_path}")
        return

    success, landmarks = facemark.fit(gray, faces)
    if not success:
        return

    keypoints = landmarks[0][0].reshape(-1)
    all_keypoints.append(keypoints)
    all_image_paths.append(image_path)

    # 绘制关键点图（白底 + 黑点）
    canvas = np.ones((gray.shape[0], gray.shape[1], 3), dtype=np.uint8) * 255
    for (x, y) in landmarks[0][0]:
        cv2.circle(canvas, (int(x), int(y)), 1, (0, 0, 0), -1)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, canvas)

    success_count[class_name] += 1

def batch_convert_dataset(input_root, output_root):
    valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    for root, _, files in os.walk(input_root):
        for fname in files:
            if not any(fname.lower().endswith(ext) for ext in valid_extensions):
                continue
            input_path = os.path.join(root, fname)
            rel_path = os.path.relpath(input_path, input_root)
            output_path = os.path.join(output_root, rel_path)

            class_name = rel_path.split(os.sep)[0]
            process_image_to_keypoints_canvas(input_path, output_path, class_name)

if __name__ == "__main__":
    input_dir = "facial_expression_dataset/train"
    output_dir = "facial_expression_dataset_lbf_pca/train"
    batch_convert_dataset(input_dir, output_dir)

    print("\n关键点检测统计结果:")
    for cls in sorted(total_count.keys()):
        total = total_count[cls]
        success = success_count[cls]
        print(f"类 [{cls}]: 总数 = {total}，成功检测 = {success}，成功率 = {success / total:.2%}")

    # ========== PCA 降维 ==========
    if len(all_keypoints) == 0:
        print("未提取到任何关键点，跳过 PCA。")


    all_keypoints = np.array(all_keypoints)
    print(f"\n执行 PCA 降维，输入数据形状: {all_keypoints.shape}")

    pca = PCA(n_components=30)
    X_pca = pca.fit_transform(all_keypoints)
    X_recon = pca.inverse_transform(X_pca)

    # 可视化 PCA 主成分（前3个）
    pca_output_dir = os.path.join(output_dir, "pca_components")
    os.makedirs(pca_output_dir, exist_ok=True)

    for i in range(3):
        component = pca.components_[i]
        if component.shape[0] == 136:
            kp = component.reshape(68, 2)
            plt.figure(figsize=(5, 5))
            plt.xlim(-0.1, 0.1)
            plt.ylim(0.1, -0.1)
            plt.gca().set_facecolor("white")
            plt.scatter(kp[:, 0], kp[:, 1], c="black", s=10)
            plt.title(f"PCA Component {i+1}")
            plt.axis("off")
            plt.savefig(os.path.join(pca_output_dir, f"component_{i+1}.png"), bbox_inches="tight")
            plt.close()

    # 可视化重建图（原图 vs PCA还原关键点图）
    recon_dir = os.path.join(output_dir, "pca_reconstruction")
    os.makedirs(recon_dir, exist_ok=True)

    for i in range(min(5, len(all_image_paths))):  # 前5张可视化
        original_kp = all_keypoints[i].reshape(68, 2)
        recon_kp = X_recon[i].reshape(68, 2)

        # 原图关键点
        plt.figure(figsize=(5, 5))
        plt.xlim(0, 48)
        plt.ylim(48, 0)
        plt.axis("off")
        plt.gca().set_facecolor("white")
        plt.scatter(original_kp[:, 0]*24+24, original_kp[:, 1]*24+24, c="black", s=10)
        plt.title("Original Keypoints")
        plt.savefig(os.path.join(recon_dir, f"sample_{i}_original.png"), bbox_inches="tight")
        plt.close()

        # 重建关键点
        plt.figure(figsize=(5, 5))
        plt.xlim(0, 48)
        plt.ylim(48, 0)
        plt.axis("off")
        plt.gca().set_facecolor("white")
        plt.scatter(recon_kp[:, 0]*24+24, recon_kp[:, 1]*24+24, c="black", s=10)
        plt.title("PCA Reconstructed")
        plt.savefig(os.path.join(recon_dir, f"sample_{i}_reconstructed.png"), bbox_inches="tight")
        plt.close()

    print(f"\nPCA主成分和重建图像已保存至: {pca_output_dir} 和 {recon_dir}")
