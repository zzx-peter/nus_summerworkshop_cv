# Import necessary libraries
import sys
import os
import cv2
import numpy as np
import threading
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
from ultralytics import YOLO
from score_engine import ScoreEngine, score_to_label
import time  

# Load YOLOv8 pose model
# This will be the starting model to extract poses from videos.
# However, you can replace it with any other pose model compatible with YOLOv8.
model = YOLO("yolov8n-pose.pt")

# COCO skeleton connections (keypoint pairs)
# This defines the connections between keypoints for drawing the skeleton.
# For better drawing of the skeleton, you can modify the pairs.
skeleton = [
    (0, 5), (0, 6),     # noise to shoulders
    (5, 6),             # shoulders
    (5, 7), (7, 9),     # left arm
    (6, 8), (8, 10),    # right arm
    (5, 11), (6, 12),   # torso sides
    (11, 12),           # hips
    (11, 13), (13, 15), # left leg
    (12, 14), (14, 16)  # right leg
]


class PoseApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Stickman Dance GUI")
        self.root.geometry("1200x600") # You can adjust the window size as needed

        self.running_file = False
        self.running_cam = False
        self.video_path = ""
        self.cap_file = None
        self.cap_cam = None
        self.show_video_frame = True

        self.fps = 15.0  # 必须和评分器一致
        self.scorer = ScoreEngine(
            beat_audio_path="",
            sample_rate=self.fps,
            use_beat=False)
        self.current_frame_index = 0


        # Set up frames
        self.left_frame = tk.Frame(self.root)
        self.left_frame.pack(side=tk.LEFT, padx=10)
        self.right_frame = tk.Frame(self.root)
        self.right_frame.pack(side=tk.RIGHT, padx=10)

        # Video File Window
        self.label_file = tk.Label(self.left_frame)
        self.label_file.pack()
        self.controls_file = tk.Frame(self.left_frame)
        self.controls_file.pack()
        tk.Button(self.controls_file, text="Open Video", command=self.load_video).pack(side=tk.LEFT, padx=5)
        tk.Button(self.controls_file, text="Start Video", command=self.start_video).pack(side=tk.LEFT, padx=5)
        tk.Button(self.controls_file, text="Stop Video", command=self.stop_video).pack(side=tk.LEFT, padx=5)
        tk.Button(self.controls_file, text="Show/Hide Video", command=self.toggle_video_display).pack(side=tk.LEFT, padx=5)

        # Webcam Window
        self.label_cam = tk.Label(self.right_frame)
        self.label_cam.pack()
        self.controls_cam = tk.Frame(self.right_frame)
        self.controls_cam.pack()
        tk.Button(self.controls_cam, text="Start Webcam", command=self.start_cam).pack(side=tk.LEFT, padx=5)
        tk.Button(self.controls_cam, text="Start Dance", command=self.start_match).pack(side=tk.LEFT, padx=5)
        tk.Button(self.controls_cam, text="Stop Webcam", command=self.stop_cam).pack(side=tk.LEFT, padx=5)

    def load_video(self):
        path = filedialog.askopenfilename(filetypes=[("MP4 files", "*.mp4"), ("All files", "*.*")])
        if path:
            self.video_path = path
            messagebox.showinfo("Video Selected", os.path.basename(path))
            # 重新加载评分器，使用选中的视频提取节拍
            print(f"[INFO] 加载参考视频用于打分: {path}")
            self.scorer = ScoreEngine(beat_audio_path=path, sample_rate=self.fps, use_beat=False)  # ★ 新增

    def start_video(self):
        if not self.video_path:
            messagebox.showwarning("No Video", "Please select a video first.")
            return
        if not self.running_file:
            self.running_file = True
            threading.Thread(target=self.process_video_file, daemon=True).start()

    def stop_video(self):
        self.running_file = False
        if self.cap_file:
            self.cap_file.release()
    
    def toggle_video_display(self):
        self.show_video_frame = not self.show_video_frame

    def start_cam(self):
        if not self.running_cam:
            self.running_cam = True
            threading.Thread(target=self.process_webcam, daemon=True).start()

    def start_match(self):
        if not self.video_path:
            messagebox.showwarning("No Video", "请先选择参考视频")
            return
        if not self.running_cam:
            self.running_cam = True
            threading.Thread(target=self.process_dance_match, daemon=True).start()

    def stop_cam(self):
        self.running_cam = False
        if self.cap_cam:
            self.cap_cam.release()

    def process_video_file(self):
        self.cap_file = cv2.VideoCapture(self.video_path)
        while self.cap_file.isOpened() and self.running_file:
            ret, frame = self.cap_file.read()
            if not ret:
                break
            frame, _ = self.process_pose(frame)
            self.update_label(self.label_file, frame)
        self.cap_file.release()

    def process_webcam(self):
        self.cap_cam = cv2.VideoCapture(0)
        while self.cap_cam.isOpened() and self.running_cam:
            ret, frame = self.cap_cam.read()
            if not ret:
                break
            # —— 在这里做水平镜像，让画面与现实方向一致 ——
            frame = cv2.flip(frame, 1)
            frame, _ = self.process_pose(frame)
            self.update_label(self.label_cam, frame)
        self.cap_cam.release()

    # —— 对舞评分核心 —— ★
    def process_dance_match(self):    
        self.cap_cam  = cv2.VideoCapture(0)
        self.cap_file = cv2.VideoCapture(self.video_path)

        while (self.cap_cam.isOpened() and self.cap_file.isOpened()
               and self.running_cam):
            ret_cam,  frame_cam  = self.cap_cam.read()
            ret_ref,  frame_ref  = self.cap_file.read()
            if not ret_cam or not ret_ref: break

            frame_cam = cv2.flip(frame_cam, 1)

            # 提取关键点与可视化
            cam_vis,  cam_kpts  = self.process_pose(frame_cam)
            ref_vis,  ref_kpts  = self.process_pose(frame_ref)

            # 评分
            if cam_kpts is not None and ref_kpts is not None:
                if self.current_frame_index % 2 == 0:  
                    print(f"[DEBUG] 打分中: 帧 {self.current_frame_index}")
                    t = self.current_frame_index / self.fps
                    score, _ = self.scorer.update(ref_kpts, cam_kpts, t)
                    label = score_to_label(score)
                    cv2.putText(cam_vis, f"{label} {score:.0%}", (30,60),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 3)

            # # ---- 交由主线程刷新 GUI ----
            # self.root.after(0, self.update_label, self.label_cam,  cam_vis)
            # self.root.after(0, self.update_label, self.label_file, ref_vis)

            # 更新 GUI
            self.update_label(self.label_cam,  cam_vis)
            # self.update_label(self.label_file, ref_vis)

            self.current_frame_index += 1
            time.sleep(1 / self.fps)   # 限制读取/评分速度

        self.cap_cam.release()
        self.cap_file.release()

    def process_pose(self, frame):
        keypoints_ndarray = None  
        results = model(frame, conf=0.3)
        height, width = frame.shape[:2]

        if self.show_video_frame:
            overlay = frame.copy()
        else:
            overlay = np.ones_like(frame) * 255  # white background

        for result in results:
            if result.keypoints is not None:
                keypoints_xyn = result.keypoints.xyn.cpu().numpy()
                for idx, person_kpts in enumerate(keypoints_xyn):
                    keypoints = [
                        (int(x * width), int(y * height)) for x, y in person_kpts
                    ]

                    for pt1, pt2 in skeleton:
                        if pt1 < len(keypoints) and pt2 < len(keypoints):
                            x1, y1 = keypoints[pt1]
                            x2, y2 = keypoints[pt2]
                            cv2.line(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2) # green lines for skeleton

                    for x, y in keypoints:
                        cv2.circle(overlay, (x, y), 5, (0, 0, 255), -1) # red circles for keypoints

                    # 将当前帧第一个人的关键点保存为浮点二维数组
                    if idx == 0:
                        keypoints_ndarray = np.array([[x * width, y * height] for x, y in person_kpts], dtype=np.float32)
                        break  # 只处理第一个人
        return overlay, keypoints_ndarray

    def update_label(self, label, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        img = img.resize((640, 384))  # Resize for better fit
        imgtk = ImageTk.PhotoImage(image=img)
        label.imgtk = imgtk
        label.configure(image=imgtk)

if __name__ == "__main__":
    root = tk.Tk()
    app = PoseApp(root)
    root.mainloop()