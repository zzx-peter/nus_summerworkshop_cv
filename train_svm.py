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
    def __init__(self, root_dir, limit_per_class=120):  # 限制每类最大数量
        self.samples = []
        self.labels = []
        self.class_to_idx = {cls: idx for idx, cls in enumerate(CLASSES)}

        self.face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        self.facemark = cv2.face.createFacemarkLBF()
        self.facemark.loadModel('lbfmodel.yaml')

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

    def get_data(self):
        return np.array(self.samples), np.array(self.labels)

def train_svm():
    print("加载训练集...")
    train_set = LandmarkDataset('facial_expression_dataset_contrast/train')
    X_train, y_train = train_set.get_data()

    print("加载测试集...")
    test_set = LandmarkDataset('facial_expression_dataset_contrast/test')
    X_test, y_test = test_set.get_data()

    print("开始训练 SVM...")
    clf = SVC(kernel='rbf', C=10, class_weight='balanced')
    clf.fit(X_train, y_train)

    os.makedirs("models", exist_ok=True)
    joblib.dump(clf, "models/expression.pkl")
    print("模型保存为 models/expression.pkl")

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
