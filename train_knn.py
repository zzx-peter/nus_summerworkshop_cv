# train_expression_model_knn.py
import os, cv2, joblib, mediapipe as mp
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from expression_classifier_knn import CLASSES   # 复用同一常量列表

# -------- 数据集加载：与 RF 版本完全一致 --------
class LandmarkDataset:
    def __init__(self, root_dir):
        self.samples, self.labels = [], []
        idx_map = {c: i for i, c in enumerate(CLASSES)}

        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        facemark = cv2.face.createFacemarkLBF()
        facemark.loadModel('lbfmodel.yaml')

        for cls in CLASSES:
            cls_dir = os.path.join(root_dir, cls)
            for fname in os.listdir(cls_dir):
                img = cv2.imread(os.path.join(cls_dir, fname), cv2.IMREAD_GRAYSCALE)
                if img is None:                                 continue
                faces = face_cascade.detectMultiScale(img, 1.1, 5)
                if len(faces) == 0:                             continue
                ok, lm = facemark.fit(img, faces)
                if not ok:                                      continue

                pts = lm[0][0].reshape(-1)
                pts = (pts - np.mean(pts)) / np.std(pts)        # z-score
                self.samples.append(pts.astype(np.float32))
                self.labels.append(idx_map[cls])

    def get_data(self):
        return np.array(self.samples), np.array(self.labels)


def train_knn():
    print('载入数据 …')
    X_train, y_train = LandmarkDataset('facial_expression_dataset_contrast/train').get_data()
    X_test,  y_test  = LandmarkDataset('facial_expression_dataset_contrast/test').get_data()

    # -------- KNN 模型 --------
    print('训练 KNN …')
    knn = KNeighborsClassifier(
        n_neighbors=5,       # K 值，可改 3~9 做网格搜索
        weights='distance',  # 距离加权投票：近样本影响更大
        algorithm='auto',    # sklearn 自动选择 KD-Tree / Ball-Tree / brute
        n_jobs=-1            # 启用多线程加速
    )
    knn.fit(X_train, y_train)

    # -------- 保存模型 --------
    os.makedirs('models', exist_ok=True)
    joblib.dump(knn, 'models/expression_knn.pkl')
    print('模型保存至 models/expression_knn.pkl')

    # -------- 评估 --------
    y_pred = knn.predict(X_test)
    print('\n测试集报告:')
    print(classification_report(y_test, y_pred, target_names=CLASSES, digits=4))
    print('混淆矩阵:')
    print(confusion_matrix(y_test, y_pred))

    print(f'训练准确率: {accuracy_score(y_train, knn.predict(X_train)):.4f}')
    print(f'测试准确率:  {accuracy_score(y_test,  y_pred):.4f}')


if __name__ == '__main__':
    train_knn()
