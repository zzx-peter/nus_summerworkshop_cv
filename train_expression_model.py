import os
import cv2
import torch
import numpy as np
from expression_classifier import ExpressionMLP, CLASSES
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix 

# ======= 数据集类，用于提取68个关键点 =======
class LandmarkDataset(Dataset):
    def __init__(self, root_dir):
        self.samples = []
        self.labels = []
        self.class_to_idx = {cls: idx for idx, cls in enumerate(CLASSES)}

        # 加载人脸检测器和关键点模型
        self.face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        self.facemark = cv2.face.createFacemarkLBF()
        self.facemark.loadModel('lbfmodel.yaml')

        for class_name in CLASSES:
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            for img_name in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_name)
                image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if image is None:
                    continue

                faces = self.face_cascade.detectMultiScale(image, 1.1, 5)
                if len(faces) == 0:
                    continue

                success, landmarks = self.facemark.fit(image, faces)
                if not success:
                    continue

                # 使用第一个人脸的关键点
                keypoints = landmarks[0][0].reshape(-1)
                keypoints = (keypoints - np.mean(keypoints)) / np.std(keypoints)
                self.samples.append(keypoints.astype(np.float32))
                self.labels.append(self.class_to_idx[class_name])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return torch.tensor(self.samples[idx]), torch.tensor(self.labels[idx])

def train(model, train_loader, test_loader, device, save_path):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(1, 21):
        model.train()
        total_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # 计算测试准确率
        model.eval()
        correct, total = 0, 0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                preds = torch.argmax(logits, dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y.cpu().numpy())

        acc = correct / total
        print(f"[Epoch {epoch:02d}] Train Loss: {total_loss:.4f} | Test Accuracy: {acc:.4f}")

    # 保存模型
    torch.save(model.state_dict(), save_path)
    print(f"✅ 模型已保存至 {save_path}")

    # 打印最终报告
    print("\n📊 最终分类报告:")
    print(classification_report(all_labels, all_preds, target_names=CLASSES, digits=4))
    print("🧮 混淆矩阵:")
    print(confusion_matrix(all_labels, all_preds))

# ======= 主函数入口 =======
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_set = LandmarkDataset('facial_expression_dataset/train')
    test_set = LandmarkDataset('facial_expression_dataset/test')

    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=64, shuffle=False)

    model = ExpressionMLP().to(device)
    train(model, train_loader, test_loader, device, save_path='models/expression_mlp.pth')