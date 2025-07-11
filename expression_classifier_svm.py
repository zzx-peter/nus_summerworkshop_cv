# expression_classifier_svm.py
import numpy as np
import joblib

CLASSES = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

class ExpressionClassifier:
    def __init__(self, model_path='models/expression_svm.pkl'):
        self.model = joblib.load(model_path)

    def predict(self, keypoints_68):
        # 输入为 68x2 的关键点，先扁平化并归一化
        pts = keypoints_68.reshape(-1)
        pts = (pts - np.mean(pts)) / np.std(pts)
        pred = self.model.predict([pts])[0]
        return CLASSES[pred], None


# 示例
if __name__ == '__main__':
    clf = ExpressionClassifier()
    dummy_keypoints = np.random.rand(68, 2) * 100
    label, _ = clf.predict(dummy_keypoints)
    print("预测结果:", label)
