# expression_classifier_knn.py
import numpy as np
import joblib

CLASSES = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

class ExpressionClassifier:
    """
    读取训练好的 KNN pkl，并提供 predict(keypoints_68) 接口
    """
    def __init__(self, model_path='models/expression_knn.pkl'):
        self.model = joblib.load(model_path)

    def predict(self, keypoints_68):
        # 68x2 -> 136 向量 + z-score 标准化（与训练时一致）
        pts = keypoints_68.reshape(-1)
        pts = (pts - np.mean(pts)) / np.std(pts)
        pred = self.model.predict([pts])[0]
        return CLASSES[pred], None      # 第二个返回值占位，与 RF/SVM 接口一致


# --- 快速自检 ---
if __name__ == '__main__':
    clf = ExpressionClassifier()
    dummy_pts = np.random.rand(68, 2) * 100
    label, _ = clf.predict(dummy_pts)
    print('预测结果:', label)
