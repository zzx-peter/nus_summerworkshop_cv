import cv2
# from expression_classifier import ExpressionClassifier
from expression_classifier_rf import ExpressionClassifier

# 加载 Haar Cascade 人脸检测模型
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# 创建 LBF 人脸标记检测器
facemark = cv2.face.createFacemarkLBF()
facemark.loadModel('lbfmodel.yaml')  # 加载68个关键点的预训练模型

# 初始化表情识别模型
# classifier = ExpressionClassifier(model_path='models/expression_mlp.pth', device='cuda' if cv2.ocl.haveOpenCL() else 'cpu')
classifier = ExpressionClassifier(model_path='models/expression_rf.pkl')
# Open webcam
cap = cv2.VideoCapture(0)
cv2.namedWindow('Face Keypoints', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Face Keypoints', 640, 480)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # 将图像转换为灰度图（人脸检测要求）
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 检测人脸（返回值是一个人脸矩形框列表）
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    
    # 如果检测到人脸，则进行关键点检测
    if len(faces) > 0:
        _, landmarks = facemark.fit(gray, faces)

        # 绘制人脸框和关键点
        for rect, landmark in zip(faces, landmarks):
            (x, y, w, h) = rect
            # 画人脸矩形框
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            # landmark[0] 是68个关键点的坐标
            for (x_point, y_point) in landmark[0]:
                cv2.circle(frame, (int(x_point), int(y_point)), 2, (0, 0, 255), -1)
            
            # 表情分类
            expression, _ = classifier.predict(landmark[0])
            # 显示分类结果
            cv2.putText(frame, expression, (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            cv2.imshow('Face Keypoints', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
