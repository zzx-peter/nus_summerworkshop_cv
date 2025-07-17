# train_rf.py

import os
import cv2
import numpy as np
import joblib
import shutil
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.decomposition import PCA
from expression_classifier_rf import CLASSES
import face_alignment
import torch

# 精选语义点对列表（基于 LBF/FAN 标准68点）
SELECTED_PAIRS = [
    (21, 22),  # 两眉间距
    (19, 37),  # 左眉-左眼
    (24, 44),  # 右眉-右眼
    (30, 48),  # 鼻尖-左嘴角
    (30, 54),  # 鼻尖-右嘴角
    (39, 42),  # 眼间距
    (48, 54),  # 嘴宽
    (51, 57),  # 嘴唇上下
    (17, 21),  # 左眉横向张开
    (22, 26),  # 右眉横向张开
    (27, 30),  # 鼻梁长度
    (36, 45),  # 左右眼外角
    (31, 35),  # 鼻翼宽度
]

def extract_distance_features(keypoints_1d):
    landmarks = keypoints_1d.reshape((-1, 2))
    features = []
    for i, j in SELECTED_PAIRS:
        dist = np.linalg.norm(landmarks[i] - landmarks[j])
        features.append(dist)
    return np.array(features, dtype=np.float32)

# ==== 1. 使用 OpenCV-LBF 提取关键点并转换为距离特征 ====
class LandmarkDataset:
    def __init__(self, root_dir, cache_file=None, limit_per_class=120):
        self.samples = []
        self.labels = []
        self.class_to_idx = {cls: idx for idx, cls in enumerate(CLASSES)}

        if cache_file and os.path.exists(cache_file):
            print(f"读取关键点缓存: {cache_file}")
            data = np.load(cache_file)
            self.samples = data["samples"]
            self.labels = data["labels"]
            return

        print("缓存未找到，开始提取关键点 ...")
        self.face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
        self.facemark = cv2.face.createFacemarkLBF()
        self.facemark.loadModel("lbfmodel.yaml")

        for class_name in CLASSES:
            class_dir = os.path.join(root_dir, class_name)
            used = 0
            for fname in os.listdir(class_dir):
                path = os.path.join(class_dir, fname)
                image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                if image is None:
                    continue
                faces = self.face_cascade.detectMultiScale(image, 1.1, 5)
                if len(faces) == 0:
                    continue
                success, landmarks = self.facemark.fit(image, faces)
                if not success:
                    continue

                keypoints = landmarks[0][0].reshape(-1)
                keypoints = (keypoints - np.mean(keypoints)) / np.std(keypoints)
                features = extract_distance_features(keypoints)
                self.samples.append(features)
                self.labels.append(self.class_to_idx[class_name])
                used += 1

        self.samples = np.array(self.samples)
        self.labels = np.array(self.labels)

        if cache_file:
            os.makedirs(os.path.dirname(cache_file), exist_ok=True)
            np.savez_compressed(cache_file, samples=self.samples, labels=self.labels)
            print(f"关键点提取完成，已缓存到: {cache_file}")

    def get_data(self):
        return self.samples, self.labels

# ==== 2. 使用 FAN 提取关键点并转换为距离特征 ====
class LandmarkDatasetFAN:
    def __init__(self, root_dir):
        self.samples = []
        self.labels = []
        self.class_to_idx = {cls: idx for idx, cls in enumerate(CLASSES)}
        self.fa = face_alignment.FaceAlignment(
            face_alignment.LandmarksType.TWO_D,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )

        for class_name in CLASSES:
            class_dir = os.path.join(root_dir, class_name)
            for fname in os.listdir(class_dir):
                path = os.path.join(class_dir, fname)
                image_bgr = cv2.imread(path)
                if image_bgr is None:
                    continue
                image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
                preds = self.fa.get_landmarks(image_rgb)
                if not preds:
                    continue
                keypoints = preds[0].reshape(-1)
                keypoints = (keypoints - np.mean(keypoints)) / np.std(keypoints)
                features = extract_distance_features(keypoints)
                self.samples.append(features)
                self.labels.append(self.class_to_idx[class_name])

    def get_data(self):
        return np.array(self.samples), np.array(self.labels)

# ==== 3. 使用整张灰度图（备用，不涉及关键点） ====
class ImageDatasetRaw:
    def __init__(self, root_dir):
        self.samples = []
        self.labels = []
        self.class_to_idx = {cls: idx for idx, cls in enumerate(CLASSES)}

        for class_name in CLASSES:
            class_dir = os.path.join(root_dir, class_name)
            for fname in os.listdir(class_dir):
                path = os.path.join(class_dir, fname)
                image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                if image is None or image.shape != (48, 48):
                    continue
                flat = image.flatten().astype(np.float32)
                flat = flat / 255.0
                self.samples.append(flat)
                self.labels.append(self.class_to_idx[class_name])

    def get_data(self):
        return np.array(self.samples), np.array(self.labels)

# ==== 训练主流程 ====
def train_rf():
    cache_dir = "cache"
    os.makedirs(cache_dir, exist_ok=True)
    train_cache = os.path.join(cache_dir, "train_landmarks.npz")
    test_cache = os.path.join(cache_dir, "test_landmarks.npz")

    print("加载训练集 ...")
    train_set = LandmarkDataset(
        root_dir="facial_expression_dataset/train", cache_file=train_cache
    )
    X_train, y_train = train_set.get_data()

    print("加载测试集 ...")
    test_set = LandmarkDataset(
        root_dir="facial_expression_dataset/test", cache_file=test_cache
    )
    X_test, y_test = test_set.get_data()

    # print("使用 PCA 降维 ...")
    # pca = PCA(n_components=50)
    # X_train = pca.fit_transform(X_train)
    # X_test = pca.transform(X_test)

    print("开始训练 RandomForest ...")
    clf = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
    )
    clf.fit(X_train, y_train)

    print("\n测试集评估:")
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=CLASSES, digits=4))
    print("混淆矩阵:")
    print(confusion_matrix(y_test, y_pred))

    train_acc = accuracy_score(y_train, clf.predict(X_train))
    test_acc = accuracy_score(y_test, y_pred)

    print(f"训练集准确率: {train_acc:.4f}")
    print(f"测试集准确率: {test_acc:.4f}")

if __name__ == "__main__":
    train_rf()
