import os
import cv2

# 输入路径和输出路径
input_root = 'facial_expression_dataset'
output_root = 'facial_expression_dataset_contrast'

# 创建输出文件夹结构
for split in ['train', 'test']:
    for emotion in os.listdir(os.path.join(input_root, split)):
        os.makedirs(os.path.join(output_root, split, emotion), exist_ok=True)

# 图像增强函数：使用直方图均衡化
def enhance_contrast(image):
    return cv2.equalizeHist(image)

# 遍历数据集并增强
for split in ['train', 'test']:
    for emotion in os.listdir(os.path.join(input_root, split)):
        input_dir = os.path.join(input_root, split, emotion)
        output_dir = os.path.join(output_root, split, emotion)

        for fname in os.listdir(input_dir):
            input_path = os.path.join(input_dir, fname)
            output_path = os.path.join(output_dir, fname)

            # 读取灰度图
            img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"跳过无法读取的文件: {input_path}")
                continue

            # 对比度增强
            enhanced = enhance_contrast(img)

            # 保存增强后的图像
            cv2.imwrite(output_path, enhanced)

print("✅ 所有图像对比度增强完成并保存至:", output_root)
