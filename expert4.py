import cv2
import os
import torch
import face_alignment
import numpy as np

# 初始化 face_alignment 模型（68关键点检测）
device = 'cuda' if torch.cuda.is_available() else 'cpu'
fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, device=device)


# 对比度增强函数（保持不变）
def adjust_contrast(image, contrast_factor=1.5):
    """增强图像对比度，支持灰度图和彩色图"""
    if len(image.shape) == 3:  # 彩色图
        ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        channels = list(cv2.split(ycrcb))
        channels[0] = cv2.convertScaleAbs(channels[0], alpha=contrast_factor, beta=128*(1-contrast_factor))
        ycrcb = cv2.merge(channels)
        return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
    else:  # 灰度图
        return cv2.convertScaleAbs(image, alpha=contrast_factor, beta=128*(1-contrast_factor))


# 关键点连线函数（保持不变）
def draw_facial_landmarks(image, landmarks):
    """绘制面部关键点连线（68点标准划分）"""
    landmarks = landmarks.astype(int)
    face_contour = list(range(0, 17))
    left_eyebrow = list(range(17, 22))
    right_eyebrow = list(range(22, 27))
    nose_bridge = list(range(27, 36))
    left_eye = list(range(36, 42))
    right_eye = list(range(42, 48))
    mouth_outline = list(range(48, 61))
    mouth_inner = list(range(61, 68))

    regions = [
        (face_contour, (0, 255, 0), 1),
        (left_eyebrow, (0, 128, 255), 1),
        (right_eyebrow, (0, 128, 255), 1),
        (nose_bridge, (255, 0, 0), 1),
        (left_eye, (0, 0, 255), 1),
        (right_eye, (0, 0, 255), 1),
        (mouth_outline, (255, 0, 255), 1),
        (mouth_inner, (255, 0, 255), 1),
    ]

    for points_idx, color, thickness in regions:
        for i in range(len(points_idx) - 1):
            start = landmarks[points_idx[i]]
            end = landmarks[points_idx[i + 1]]
            cv2.line(image, tuple(start), tuple(end), color, thickness)

    # 闭合眼睛区域
    cv2.line(image, tuple(landmarks[left_eye[-1]]), tuple(landmarks[left_eye[0]]), (0, 0, 255), 1)
    cv2.line(image, tuple(landmarks[right_eye[-1]]), tuple(landmarks[right_eye[0]]), (0, 0, 255), 1)

    return image


# 主函数：处理单个图片文件（核心替换部分）
def process_image(image_path, output_dir=None):
    """处理单张图片：对比度增强→人脸检测→关键点连线→保存结果"""
    # 加载图片
    image = cv2.imread(image_path)
    if image is None:
        print(f"图片加载失败：{image_path}")
        return None

    # 调整对比度
    image_contrast = adjust_contrast(image)

    # 转换为 RGB 格式（face_alignment 要求输入为 RGB）
    image_rgb = cv2.cvtColor(image_contrast, cv2.COLOR_BGR2RGB)
    
    # 使用 face_alignment 检测关键点（核心替换）
    landmarks_list = fa.get_landmarks(image_rgb)  # 返回列表，每个元素是一张人脸的68点坐标

    result = np.ones_like(image_contrast) * 255  # 白底图
    if landmarks_list is not None and len(landmarks_list) > 0:
        # 取第一张检测到的人脸
        landmarks = np.array(landmarks_list[0])
        
        # 绘制人脸框（根据关键点计算最小外接矩形）
        min_x, min_y = np.min(landmarks, axis=0).astype(int)
        max_x, max_y = np.max(landmarks, axis=0).astype(int)
        cv2.rectangle(result, (min_x, min_y), (max_x, max_y), (0, 255, 0), 2)
        
        # 绘制线稿
        result = draw_facial_landmarks(result, landmarks)
        print(f"成功处理：{image_path}，检测到 {len(landmarks_list)} 张人脸")
    else:
        print(f"警告：{image_path} 未检测到人脸")

    # 保存结果
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, os.path.basename(image_path))
        cv2.imwrite(output_path, result)
        print(f"已保存结果到：{output_path}")

    return result


# 主程序：遍历目录处理所有图片（保持不变）
def batch_process_images(input_dir, output_dir=None):
    """批量处理目录中的所有图片"""
    valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    
    # 如果输入是文件，直接处理单个文件
    if os.path.isfile(input_dir):
        if any(input_dir.lower().endswith(ext) for ext in valid_extensions):
            process_image(input_dir, output_dir)
        else:
            print(f"错误：{input_dir} 不是有效图片文件")
        return

    # 遍历目录处理所有图片
    for root, _, files in os.walk(input_dir):
        for file in files:
            if any(file.lower().endswith(ext) for ext in valid_extensions):
                file_path = os.path.join(root, file)
                # 构建输出路径
                if output_dir:
                    relative_path = os.path.relpath(root, input_dir)
                    out_subdir = os.path.join(output_dir, relative_path)
                else:
                    out_subdir = None
                process_image(file_path, out_subdir)


if __name__ == "__main__":
    # 配置输入输出路径
    input_path = 'facial_expression_dataset/test'  # 替换为你的输入路径
    output_path = 'facial_expression_dataset/test_lineart'  # 替换为你的输出路径
    # 执行批量处理
    batch_process_images(input_path, output_path)