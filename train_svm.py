# train_expression_model_svm.py
import os
import cv2
import numpy as np
import joblib
import mediapipe as mp
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from expression_classifier_svm import CLASSES

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

def train_svm():
    # 缓存文件路径
    cache_dir = "cache"
    os.makedirs(cache_dir, exist_ok=True)
    train_cache = os.path.join(cache_dir, "train_landmarks.npz")
    test_cache = os.path.join(cache_dir, "test_landmarks.npz")

    print("加载训练集 ...")
    train_set = LandmarkDataset(
        root_dir="facial_expression_dataset/train"
    )
    X_train, y_train = train_set.get_data()

    print("加载测试集 ...")
    test_set = LandmarkDataset(
        root_dir="facial_expression_dataset/test"
    )
    X_test, y_test = test_set.get_data()

    print("开始训练 SVM...")
    clf = SVC(
        kernel='rbf',
        C=5,
        # class_weight='balanced'
    )
    clf.fit(X_train, y_train)

    os.makedirs("models", exist_ok=True)
    joblib.dump(clf, "models/expression_svm.pkl")
    print("模型保存为 models/expression_svm.pkl")

    print("\n测试集评估:")
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=CLASSES, digits=4))
    print("混淆矩阵:")
    print(confusion_matrix(y_test, y_pred))

    train_acc = accuracy_score(y_train, clf.predict(X_train))
    test_acc = accuracy_score(y_test, y_pred)
    print(f"训练集准确率: {train_acc:.4f}")
    print(f"测试集准确率: {test_acc:.4f}")

if __name__ == '__main__':
    train_svm()
