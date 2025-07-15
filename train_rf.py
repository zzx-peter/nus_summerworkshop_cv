# train_rf_cached.py
# 与原版 train_rf.py 逻辑一致，但新增了“关键点缓存”机制，
# 只在首次运行时执行人脸检测与关键点提取，后续直接读取 .npz 文件，显著加速训练/测试。

import os
import cv2
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from expression_classifier_rf import CLASSES
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA


class LandmarkDataset:
    """基于 OpenCV LBF 的人脸关键点数据集。
    
    如果提供 `cache_file` 且该文件存在，则直接加载缓存，
    否则执行关键点提取并将结果保存到缓存。
    """

    def __init__(self, root_dir, cache_file=None, limit_per_class=120):  # 限制每类最大数量
        self.samples = []
        self.labels = []
        self.class_to_idx = {cls: idx for idx, cls in enumerate(CLASSES)}

        # ---- 1. 直接读取缓存 ----
        if cache_file and os.path.exists(cache_file):
            print(f"读取关键点缓存: {cache_file}")
            data = np.load(cache_file)
            self.samples = data["samples"]
            self.labels = data["labels"]
            return

        # ---- 2. 首次运行: 提取关键点并缓存 ----
        print("缓存未找到，开始提取关键点 ...")
        self.face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
        self.facemark = cv2.face.createFacemarkLBF()
        self.facemark.loadModel("lbfmodel.yaml")

        for class_name in CLASSES:
            class_dir = os.path.join(root_dir, class_name)
            used = 0
            for fname in os.listdir(class_dir):
                # if used >= limit_per_class:
                #     break
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
                self.samples.append(keypoints.astype(np.float32))
                self.labels.append(self.class_to_idx[class_name])
                used += 1

        # 转为 ndarray 以便保存
        self.samples = np.array(self.samples)
        self.labels = np.array(self.labels)

        # 保存缓存文件，使用压缩格式节省空间
        if cache_file:
            os.makedirs(os.path.dirname(cache_file), exist_ok=True)
            np.savez_compressed(cache_file, samples=self.samples, labels=self.labels)
            print(f"关键点提取完成，已缓存到: {cache_file}")

    def get_data(self):
        return self.samples, self.labels


# ---- 2. 使用 FAN (face-alignment) 提取关键点 --------------------------------
import face_alignment
import torch


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
                self.samples.append(keypoints.astype(np.float32))
                self.labels.append(self.class_to_idx[class_name])

    def get_data(self):
        return np.array(self.samples), np.array(self.labels)


# ---- 3. 使用整张灰度图（48×48） ---------------------------------------------
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


def train_rf():
    # 缓存文件路径
    # cache_dir = "cache"
    # os.makedirs(cache_dir, exist_ok=True)
    # train_cache = os.path.join(cache_dir, "train_landmarks.npz")
    # test_cache = os.path.join(cache_dir, "test_landmarks.npz")

    # print("加载训练集 ...")
    # train_set = LandmarkDataset(
    #     root_dir="facial_expression_dataset/train", cache_file=train_cache
    # )
    # X_train, y_train = train_set.get_data()

    # print("加载测试集 ...")
    # test_set = LandmarkDataset(
    #     root_dir="facial_expression_dataset/test", cache_file=test_cache
    # )
    # X_test, y_test = test_set.get_data()

    print("加载训练集 ...")
    train_set = ImageDatasetRaw(
        root_dir="facial_expression_dataset_fan/train"
    )
    X_train, y_train = train_set.get_data()

    print("加载测试集 ...")
    test_set = ImageDatasetRaw(
        root_dir="facial_expression_dataset_fan/test"
    )
    X_test, y_test = test_set.get_data()
    # # PCA 降维
    # print("使用 PCA 降维 ...")
    # pca = PCA(n_components=30)
    # X_train = pca.fit_transform(X_train)
    # X_test = pca.transform(X_test)

    print("开始训练 RandomForest ...")
    clf = RandomForestClassifier(
        n_estimators=100,
        # max_depth=15,
        # min_samples_split=10,
        # min_samples_leaf=5,
        # class_weight="balanced",
        random_state=42,
    )
    clf.fit(X_train, y_train)

    # os.makedirs("models", exist_ok=True)
    # joblib.dump(clf, "models/expression_rf.pkl")
    # print("模型保存为 models/expression_rf.pkl")

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
