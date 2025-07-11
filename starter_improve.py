import cv2
import mediapipe as mp
from expression_classifier_rf import ExpressionClassifier

# 初始化 MediaPipe Face Mesh，468个三维点
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=5, refine_landmarks=True)

# 打开摄像头
cap = cv2.VideoCapture(0)
cv2.namedWindow('Face Mesh', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Face Mesh', 640, 480)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 转为灰度图并做直方图均衡化
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    equalized = cv2.equalizeHist(gray)
    # 转回3通道以便MediaPipe处理
    equalized_bgr = cv2.cvtColor(equalized, cv2.COLOR_GRAY2BGR)

    # MediaPipe要求RGB格式
    rgb_frame = cv2.cvtColor(equalized_bgr, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            ih, iw, _ = frame.shape
            x_list = []
            y_list = []
            for lm in face_landmarks.landmark:
                x, y = int(lm.x * iw), int(lm.y * ih)
                x_list.append(x)
                y_list.append(y)
                cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)  # 红色更小的点
            x_min, x_max = min(x_list), max(x_list)
            y_min, y_max = min(y_list), max(y_list)
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

    cv2.imshow('Face Mesh', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()