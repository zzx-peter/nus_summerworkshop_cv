# CNN Classifier
import cv2
from keras.models import load_model
import numpy as np
from statistics import mode
import random
import os
import math

class EmotionEffects:
    def __init__(self, emoji_dir="emoji"):
        # 加载表情图片
        self.emoji_images = {}
        self.load_emoji_images(emoji_dir)
        
        # 特效函数映射
        self.effect_functions = {
            'happy': self.draw_confetti,
            'angry': self.draw_fire,
            'sad': self.draw_rain,
            'surprise': self.draw_surprise_effect
        }
    
    def load_emoji_images(self, emoji_dir):
        """加载所有表情图片"""
        emoji_files = {
            'happy': 'happy.png',
            'angry': 'angry.png',
            'sad': 'sad.png',
            'surprise': 'surprise.png',
            'disgust': 'disgust.png',
            'fear': 'fear.png',
            'neutral': 'neutral.png'
        }
        
        for expression, filename in emoji_files.items():
            path = os.path.join(emoji_dir, filename)
            if os.path.exists(path):
                # 读取带透明通道的PNG
                img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                if img is not None:
                    self.emoji_images[expression] = img
                else:
                    print(f"警告: 无法加载表情图片 {path}")
            else:
                print(f"警告: 表情图片不存在 {path}")
    
    def overlay_image(self, bg, fg, pos, size=(50, 50)):
        """将带透明通道的图片叠加到背景上"""
        x, y = pos
        fg = cv2.resize(fg, size)
        h, w = fg.shape[:2]

        # 如果没有 alpha 通道，补一个全不透明的 alpha 通道
        if fg.shape[2] == 3:
            alpha_channel = np.ones((h, w, 1), dtype=fg.dtype) * 255
            fg = np.concatenate([fg, alpha_channel], axis=2)

        # 限制叠加区域不越界
        x = max(0, min(x, bg.shape[1] - w))
        y = max(0, min(y, bg.shape[0] - h))

        fg_img = fg[:, :, :3]
        fg_alpha = fg[:, :, 3] / 255.0

        roi = bg[y:y+h, x:x+w]

        for c in range(3):
            roi[:, :, c] = fg_alpha * fg_img[:, :, c] + (1 - fg_alpha) * roi[:, :, c]

        return bg
    
    def draw_confetti(self, frame, rect):
        x, y, w, h = rect
        for _ in range(30):
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            pos = (random.randint(x-w, x+2*w), random.randint(y-h, y+2*h))
            cv2.circle(frame, pos, random.randint(2, 5), color, -1)
    
    def draw_fire(self, frame, face_rect):
        x, y, w, h = face_rect

        # 外层火焰
        base_points = np.array([
            [x + w//4, y - 30],
            [x + w//3, y - 70],
            [x + w//2, y - 120],      # 外层最高
            [x + 2*w//3, y - 70],
            [x + 3*w//4, y - 30],
            [x + w//2, y - 10]
        ])
        jitter = np.random.randint(-5, 5, (6, 2))
        dynamic_points = base_points + jitter
        cv2.fillPoly(frame, [dynamic_points], (255, 0, 0))  # 外层红色

        # 中层火焰
        mid_points = dynamic_points.copy()
        mid_points[2, 1] += 20   # 顶部主火苗降低20像素
        mid_points[1:5, 1] += 10 # 两侧火苗也略降低
        mid_points[:, 0] += np.random.randint(-2, 2, 6)
        cv2.fillPoly(frame, [mid_points], (255, 100, 0))  # 中层橙色

        # 内层火焰
        inner_points = mid_points.copy()
        inner_points[2, 1] += 20   # 顶部主火苗再降低20像素
        inner_points[1:5, 1] += 8  # 两侧火苗再略降低
        inner_points[:, 0] += np.random.randint(-2, 2, 6)
        cv2.fillPoly(frame, [inner_points], (255, 200, 0))  # 内层黄色

        # 随机火星粒子
        for _ in range(random.randint(2, 5)):
            px = x + w//2 + random.randint(-15, 15)
            py = y - 120 + random.randint(-40, -10)
            size = random.randint(1, 3)
            cv2.circle(frame, (px, py), size, (255, 255, 0), -1)

        # 闪烁效果
        if random.random() < 0.1:
            bright_points = dynamic_points.copy()
            bright_points[2, 1] += 20
            cv2.fillPoly(frame, [bright_points], (255, 255, 0))

    def draw_rain(self, frame, face_rect):
        x, y, w, h = face_rect
        # 在脸部周围绘制雨滴线
        for _ in range(20):
            start = (random.randint(x, x+w), random.randint(y, y+h))
            end = (start[0]+random.randint(-5, 5), start[1]+random.randint(10, 20))
            cv2.line(frame, start, end, (0, 200, 255), 1)

    def draw_surprise_effect(self, frame, face_rect):
        x, y, w, h = face_rect
        # 绘制放射状线条（从脸部中心向外）
        top_center = (x + w//2, y+h*17//20)
        for angle in range(0, 360, 30):
            end_x = top_center[0] + int(1.5 * w * math.cos(math.radians(angle)))
            end_y = top_center[1] + int(1.5 * h * math.sin(math.radians(angle)))
            cv2.line(frame, top_center, (end_x, end_y), (255, 225, 0), 2)

    
    def apply_effect(self, frame, expression, face_rect):
        if expression in self.emoji_images:
            x, y, w, h = face_rect
            emoji_img = self.emoji_images[expression]
            self.overlay_image(frame, emoji_img, (x, y - 60))

        if expression in self.effect_functions:
            self.effect_functions[expression](frame, face_rect)

def apply_offsets(face_coordinates, offsets):
    x, y, width, height = face_coordinates
    x_off, y_off = offsets
    return (x - x_off, x + width + x_off, y - y_off, y + height + y_off)

def preprocess_input(x, v2=True):
    x = x.astype('float32')
    x = x / 255.0
    if v2:
        x = x - 0.5
        x = x * 2.0
    return x


def main():
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    emotion_classifier = load_model('models/fer2013_mini_XCEPTION.102-0.66.hdf5', compile=False)

    cap = cv2.VideoCapture(0)
    cv2.namedWindow('Face Expression', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Face Expression', 800, 600)

    effects = EmotionEffects(emoji_dir="emoji")

    emotion_offsets = (20, 40)
    emotion_labels = {0:'angry',1:'disgust',2:'fear',3:'happy',4:'sad',5:'surprise',6:'neutral'}
    emotion_history = []
    STABLE_FRAMES = 5

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        for face in faces:

            x1, x2, y1, y2 = apply_offsets(face, emotion_offsets)
            gray_face = gray[y1:y2, x1:x2]
            try:
                gray_face = cv2.resize(gray_face, emotion_classifier.input_shape[1:3])
            except:
                continue

            gray_face = preprocess_input(gray_face, True)
            gray_face = np.expand_dims(gray_face, 0)
            gray_face = np.expand_dims(gray_face, -1)
            emotion = emotion_classifier.predict(gray_face)
            emotion_probability = np.max(emotion)
            emotion_label = np.argmax(emotion)
            emotion_text = emotion_labels[emotion_label]
            emotion_history.append(emotion_text)

            if len(emotion_history) > STABLE_FRAMES:
                emotion_history.pop(0)
            try:
                emotion_mode = mode(emotion_history)
            except:
                continue
        
            last_stable_expression = None
            if len(emotion_history) == STABLE_FRAMES and len(set(emotion_history)) == 1:
                last_stable_expression = emotion_history[0]

            if emotion_text == 'angry':
                color = emotion_probability * np.asarray((255, 0, 0))
            elif emotion_text == 'sad':
                color = emotion_probability * np.asarray((0, 0, 255))
            elif emotion_text == 'happy':
                color = emotion_probability * np.asarray((255, 255, 0))
            elif emotion_text == 'surprise':
                color = emotion_probability * np.asarray((0, 255, 255))
            else:
                color = emotion_probability * np.asarray((0, 255, 0))

            color = color.astype(int)
            color = color.tolist()

            x, y, w, h = face
            cv2.rectangle(rgb, (x, y), (x+w, y+h), color, 2)
            cv2.putText(rgb, emotion_mode, (x, y-45), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 1, cv2.LINE_AA)

            if last_stable_expression:
                effects.apply_effect(rgb, last_stable_expression, (x, y, w, h))

        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        cv2.imshow('Face Expression', bgr)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

# Using LBF
# import cv2
# # from expression_classifier import ExpressionClassifier
# from expression_classifier_rf import ExpressionClassifier
# import random
# import math
# import numpy as np
# import os

# class EmotionEffects:
#     def __init__(self, emoji_dir="emoji"):
#         # 加载表情图片
#         self.emoji_images = {}
#         self.load_emoji_images(emoji_dir)
        
#         # 特效函数映射
#         self.effect_functions = {
#             'happy': self.draw_confetti,
#             'angry': self.draw_fire,
#             'sad': self.draw_rain,
#             'surprise': self.draw_surprise_effect
#         }
    
#     def load_emoji_images(self, emoji_dir):
#         """加载所有表情图片"""
#         emoji_files = {
#             'happy': 'happy.png',
#             'angry': 'angry.png',
#             'sad': 'sad.png',
#             'surprise': 'surprise.png',
#             'disgust': 'disgust.png',
#             'fear': 'fear.png',
#             'neutral': 'neutral.png'
#         }
        
#         for expression, filename in emoji_files.items():
#             path = os.path.join(emoji_dir, filename)
#             if os.path.exists(path):
#                 # 读取带透明通道的PNG
#                 img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
#                 if img is not None:
#                     self.emoji_images[expression] = img
#                 else:
#                     print(f"警告: 无法加载表情图片 {path}")
#             else:
#                 print(f"警告: 表情图片不存在 {path}")
    
#     def overlay_image(self, bg, fg, pos, size=(50, 50)):
#         """将带透明通道的图片叠加到背景上"""
#         x, y = pos
#         fg = cv2.resize(fg, size)
#         h, w = fg.shape[:2]
        
#         # 限制叠加区域不越界
#         x = max(0, min(x, bg.shape[1] - w))
#         y = max(0, min(y, bg.shape[0] - h))

#         fg_img = fg[:, :, :3]
#         fg_alpha = fg[:, :, 3] / 255.0

#         roi = bg[y:y+h, x:x+w]

#         for c in range(3):
#             roi[:, :, c] = fg_alpha * fg_img[:, :, c] + (1 - fg_alpha) * roi[:, :, c]

#         return bg
    
#     def draw_confetti(self, frame, rect):
#         x, y, w, h = rect
#         for _ in range(30):
#             color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
#             pos = (random.randint(x-w, x+2*w), random.randint(y-h, y+2*h))
#             cv2.circle(frame, pos, random.randint(2, 5), color, -1)
    
#     def draw_fire(self, frame, face_rect):
#         x, y, w, h = face_rect

#         # 外层火焰
#         base_points = np.array([
#             [x + w//4, y - 30],
#             [x + w//3, y - 70],
#             [x + w//2, y - 120],      # 外层最高
#             [x + 2*w//3, y - 70],
#             [x + 3*w//4, y - 30],
#             [x + w//2, y - 10]
#         ])
#         jitter = np.random.randint(-5, 5, (6, 2))
#         dynamic_points = base_points + jitter
#         cv2.fillPoly(frame, [dynamic_points], (0, 0, 255))  # 外层红色

#         # 中层火焰
#         mid_points = dynamic_points.copy()
#         mid_points[2, 1] += 20   # 顶部主火苗降低20像素
#         mid_points[1:5, 1] += 10 # 两侧火苗也略降低
#         mid_points[:, 0] += np.random.randint(-2, 2, 6)
#         cv2.fillPoly(frame, [mid_points], (0, 100, 255))  # 中层橙色

#         # 内层火焰
#         inner_points = mid_points.copy()
#         inner_points[2, 1] += 20   # 顶部主火苗再降低20像素
#         inner_points[1:5, 1] += 8  # 两侧火苗再略降低
#         inner_points[:, 0] += np.random.randint(-2, 2, 6)
#         cv2.fillPoly(frame, [inner_points], (0, 200, 255))  # 内层黄色

#         # 随机火星粒子
#         for _ in range(random.randint(2, 5)):
#             px = x + w//2 + random.randint(-15, 15)
#             py = y - 120 + random.randint(-40, -10)
#             size = random.randint(1, 3)
#             cv2.circle(frame, (px, py), size, (0, 255, 255), -1)

#         # 闪烁效果
#         if random.random() < 0.1:
#             bright_points = dynamic_points.copy()
#             bright_points[2, 1] += 20
#             cv2.fillPoly(frame, [bright_points], (0, 255, 255))

#     def draw_rain(self, frame, face_rect):
#         x, y, w, h = face_rect
#         # 在脸部周围绘制雨滴线
#         for _ in range(20):
#             start = (random.randint(x, x+w), random.randint(y, y+h))
#             end = (start[0]+random.randint(-5, 5), start[1]+random.randint(10, 20))
#             cv2.line(frame, start, end, (255, 200, 0), 1)

#     def draw_surprise_effect(self, frame, face_rect):
#         x, y, w, h = face_rect
#         # 绘制放射状线条（从脸部中心向外）
#         center = (x + w//2, y + h//2)
#         for angle in range(0, 360, 30):
#             end_x = center[0] + int(1.5 * w * math.cos(math.radians(angle)))
#             end_y = center[1] + int(1.5 * h * math.sin(math.radians(angle)))
#             cv2.line(frame, center, (end_x, end_y), (0, 225, 255), 2)

    
#     def apply_effect(self, frame, expression, face_rect):
#         if expression in self.emoji_images:
#             x, y, w, h = face_rect
#             emoji_img = self.emoji_images[expression]
#             self.overlay_image(frame, emoji_img, (x, y - 60))

#         if expression in self.effect_functions:
#             self.effect_functions[expression](frame, face_rect)

# def main():
#     # 初始化人脸检测和识别模型
#     face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#     facemark = cv2.face.createFacemarkLBF()
#     facemark.loadModel('lbfmodel.yaml')
#     classifier = ExpressionClassifier(model_path='models/expression_rf.pkl')
    
#     # 初始化摄像头
#     cap = cv2.VideoCapture(0)
#     cv2.namedWindow('Face Expression', cv2.WINDOW_NORMAL)
#     cv2.resizeWindow('Face Expression', 800, 600)
    
#     # 初始化表情效果系统
#     effects = EmotionEffects(emoji_dir="emoji")
    
#     # 表情历史记录
#     expression_history = []
#     STABLE_FRAMES = 5
    
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
        
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        
#         if len(faces) > 0:
#             _, landmarks = facemark.fit(gray, faces)
            
#             for rect, landmark in zip(faces, landmarks):
#                 expression, _ = classifier.predict(landmark[0])
                
#                 # 更新表情历史
#                 expression_history.append(expression)
#                 if len(expression_history) > STABLE_FRAMES:
#                     expression_history.pop(0)
                
#                 # 检查表情是否稳定
#                 last_stable_expression = None
#                 if len(expression_history) == STABLE_FRAMES and len(set(expression_history)) == 1:
#                     last_stable_expression = expression_history[0]
                
#                 # 绘制人脸框
#                 x, y, w, h = rect
#                 cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
#                 cv2.putText(frame, expression, (x+w-70, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
                
#                 # 应用稳定表情的效果
#                 if last_stable_expression:
#                     effects.apply_effect(frame, last_stable_expression, (x, y, w, h))
        
#         cv2.imshow('Face Expression', frame)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
    
#     cap.release()
#     cv2.destroyAllWindows()

# if __name__ == '__main__':

#     main()