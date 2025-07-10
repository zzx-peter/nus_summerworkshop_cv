# expression_classifier.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time

# 7 表情类别 (FER2013)：angry, disgust, fear, happy, sad, surprise, neutral
CLASSES = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']


class ExpressionMLP(nn.Module):
    def __init__(self):
        super(ExpressionMLP, self).__init__()
        self.fc1 = nn.Linear(136, 64)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(64, 7)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)  # 输出 raw logits
        return x


# ======= 推理辅助函数 =======
class ExpressionClassifier:
    def __init__(self, model_path='models/expression_mlp.pth', device='cpu'):
        self.device = torch.device(device)
        self.model = ExpressionMLP().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

    def predict(self, keypoints_68):
        """
        keypoints_68: np.array with shape (68, 2)
        """
        # 扁平化 + 归一化（中心化到 (0,0)，归一化长度）
        pts = keypoints_68.reshape(-1)
        pts = pts - np.mean(pts)  # 简单中心化
        pts = pts / np.std(pts)   # 简单标准化

        x = torch.tensor(pts, dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(x)
            prob = F.softmax(logits, dim=1)
            pred = torch.argmax(prob, dim=1).item()
            return CLASSES[pred], prob.squeeze().cpu().numpy()


# 示例用法
if __name__ == "__main__":
    clf = ExpressionClassifier()
    dummy_keypoints = np.random.rand(68, 2) * 100
    label, prob = clf.predict(dummy_keypoints)
    print("Predicted:", label)
