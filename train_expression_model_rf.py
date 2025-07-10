# train_expression_model_rf.py
import os
import cv2
import numpy as np
import joblib
import mediapipe as mp
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

# 新增 MediaPipe 数据集
class MediaPipeLandmarkDataset:
    def __init__(self, root_dir):
        self.samples = []
        self.labels = []
        self.class_to_idx = {cls: idx for idx, cls in enumerate(CLASSES)}

        self.mp_face_mesh = mp.solutions.face_mesh
        self.detector = self.mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=False)

        for class_name in CLASSES:
            class_dir = os.path.join(root_dir, class_name)
            for fname in os.listdir(class_dir):
                path = os.path.join(class_dir, fname)
                image = cv2.imread(path)
                if image is None:
                    continue
                # 灰度图转换为BGR
                if len(image.shape) == 2 or image.shape[2] == 1:  # 是灰度图
                    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                # 确保图像大小为48x48
                image = cv2.resize(image, (48, 48))
                rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = self.detector.process(rgb)
                # 检查 MediaPipe 结果是否有效，否则跳过
                if not results.multi_face_landmarks or len(results.multi_face_landmarks) == 0:
                    print(f"跳过图像 {path}，未检测到人脸或关键点")
                    continue
                landmarks = results.multi_face_landmarks[0]
                h, w, _ = image.shape
                pts = np.array([[lm.x * w, lm.y * h] for lm in landmarks.landmark], dtype=np.float32)
                pts = (pts - np.mean(pts)) / np.std(pts)
                self.samples.append(pts.flatten())
                self.labels.append(self.class_to_idx[class_name])

    def get_data(self):
        return np.array(self.samples), np.array(self.labels)

def train_rf():
    
    # print("📦 加载训练集...")
    # train_set = LandmarkDataset('facial_expression_dataset/train')
    # X_train, y_train = train_set.get_data()

    # print("📦 加载测试集...")
    # test_set = LandmarkDataset('facial_expression_dataset/test')
    # X_test, y_test = test_set.get_data()

    print("📦 加载训练集...")
    train_set = MediaPipeLandmarkDataset('facial_expression_dataset/train')
    X_train, y_train = train_set.get_data()

    print("📦 加载测试集...")
    test_set = MediaPipeLandmarkDataset('facial_expression_dataset/test')
    X_test, y_test = test_set.get_data()

    print("开始训练 RandomForest...")
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    os.makedirs("models", exist_ok=True)
    # joblib.dump(clf, "models/expression_rf.pkl")
    # print("模型保存为 models/expression_rf.pkl")
    joblib.dump(clf, "models/expression_rf_mp.pkl")
    print("模型保存为 models/expression_rf_mp.pkl")

    print("\n📊 测试集评估:")
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=CLASSES, digits=4))
    print("🧮 混淆矩阵:")
    print(confusion_matrix(y_test, y_pred))

    train_acc = accuracy_score(y_train, clf.predict(X_train))
    test_acc = accuracy_score(y_test, y_pred)

    print(f"✅ 训练集准确率: {train_acc:.4f}")
    print(f"✅ 测试集准确率: {test_acc:.4f}")


if __name__ == '__main__':
    train_rf()
