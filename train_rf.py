# train_expression_model_rf.py
import os
import cv2
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from expression_classifier_rf import CLASSES
from sklearn.metrics import accuracy_score

class LandmarkDataset:
    def __init__(self, root_dir):
        self.samples = []
        self.labels = []
        self.class_to_idx = {cls: idx for idx, cls in enumerate(CLASSES)}

        self.face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        self.facemark = cv2.face.createFacemarkLBF()
        self.facemark.loadModel('lbfmodel.yaml')

        for class_name in CLASSES:
            class_dir = os.path.join(root_dir, class_name)
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
                self.samples.append(keypoints.astype(np.float32))
                self.labels.append(self.class_to_idx[class_name])

    def get_data(self):
        return np.array(self.samples), np.array(self.labels)

# ---- 2. 使用 FAN (face-alignment) 提取关键点 --------------------------------
import face_alignment
import torch

class LandmarkDatasetFAN:
    def __init__(self, root_dir):
        self.samples = []
        self.labels = []
        self.class_to_idx = {cls: idx for idx, cls in enumerate(CLASSES)}
        self.fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, device='cuda' if torch.cuda.is_available() else 'cpu')

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
                flat = (flat - np.mean(flat)) / np.std(flat)
                self.samples.append(flat)
                self.labels.append(self.class_to_idx[class_name])

    def get_data(self):
        return np.array(self.samples), np.array(self.labels)

def train_rf():
    # print("加载训练集...")
    # train_set = LandmarkDataset('facial_expression_dataset/train')
    # X_train, y_train = train_set.get_data()

    # print("加载测试集...")
    # test_set = LandmarkDataset('facial_expression_dataset/test')
    # X_test, y_test = test_set.get_data()

    # print("加载训练集...")
    # train_set = LandmarkDatasetFAN('facial_expression_dataset/train')
    # X_train, y_train = train_set.get_data()

    # print("加载测试集...")
    # test_set = LandmarkDatasetFAN('facial_expression_dataset/test')
    # X_test, y_test = test_set.get_data()

    print("加载训练集...")
    train_set = ImageDatasetRaw('facial_expression_dataset/train')
    X_train, y_train = train_set.get_data()

    print("加载测试集...")
    test_set = ImageDatasetRaw('facial_expression_dataset/test')
    X_test, y_test = test_set.get_data()

    print("开始训练 RandomForest...")
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    os.makedirs("models", exist_ok=True)
    joblib.dump(clf, "models/expression_rf.pkl")
    print("模型保存为 models/expression_rf.pkl")

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
    train_rf()
