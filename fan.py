import os, cv2, numpy as np, torch, face_alignment
from collections import defaultdict

# ─── 初始化 FAN ────────────────────────────────────────────────────────────
device = 'cuda' if torch.cuda.is_available() else 'cpu'
fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, device=device)

# 统计字典
total_cnt   = defaultdict(int)
success_cnt = defaultdict(int)

# ─── 单张处理 ────────────────────────────────────────────────────────────────
def process_one(img_path: str, save_path: str, cls_name: str):
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        print("读取失败:", img_path); return
    total_cnt[cls_name] += 1                       # 计总数

    # FAN 需 RGB
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    lms_lst = fa.get_landmarks(img_rgb)            # None 或 [N,68,2]

    if not lms_lst:
        print("未检测到人脸:", img_path); return     # 失败→不保存

    pts = lms_lst[0].astype(int)

    # 白底画布，尺寸同原图
    canvas = np.full_like(img_bgr, 255)
    for x, y in pts:
        cv2.circle(canvas, (x, y), 1, (0, 0, 0), -1)  # 黑点

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, canvas)
    success_cnt[cls_name] += 1                      # 成功数

# ─── 批量遍历 ────────────────────────────────────────────────────────────────
def batch_convert(src_root: str, dst_root: str):
    exts = ('.jpg', '.jpeg', '.png', '.bmp')
    for root, _, files in os.walk(src_root):
        for f in files:
            if not f.lower().endswith(exts): continue
            in_path  = os.path.join(root, f)
            rel_path = os.path.relpath(in_path, src_root)
            out_path = os.path.join(dst_root, rel_path)
            cls      = rel_path.split(os.sep)[0]    # 取表情类别文件夹名
            process_one(in_path, out_path, cls)

# ─── 主函数 ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # 修改为你的数据集路径
    src_dir = "facial_expression_dataset/test"
    dst_dir = "facial_expression_dataset_fan/test"
    batch_convert(src_dir, dst_dir)

    print("\n关键点检测统计:")
    for cls in sorted(total_cnt):
        tot, suc = total_cnt[cls], success_cnt[cls]
        rate = suc / tot if tot else 0
        print(f"{cls:8s}: 总数={tot:4d}  成功={suc:4d}  成功率={rate:.2%}")
