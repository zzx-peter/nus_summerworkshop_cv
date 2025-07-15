# score_engine2.py
import json, time, math
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np

# ------------------ 打分标签 ------------------
LABELS: List[Tuple[float, str]] = [
    (0.95, "PERFECT"),
    (0.85, "GREAT"),
    (0.75, "GOOD"),
]

# ------------------ 工具函数 ------------------
def _timestamp_to_sec(t: str) -> float:
    """'H:MM:SS(.ffffff)' → 秒(float)"""
    h, m, s = t.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)

def _angle(p1, p2, p3) -> float:
    """返回 ∠p2 (弧度)。任一点缺失(NaN)返回 NaN"""
    if np.isnan(p1).any() or np.isnan(p2).any() or np.isnan(p3).any():
        return float("nan")
    v1, v2 = p1 - p2, p3 - p2
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return float("nan")
    cos = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    return math.acos(cos)

# —— COCO‑17 关键点索引 ——  
L_SHO, R_SHO = 5, 6
L_ELB, R_ELB = 7, 8
L_WRI, R_WRI = 9, 10
L_HIP, R_HIP = 11, 12
L_KNE, R_KNE = 13, 14
L_ANK, R_ANK = 15, 16

ANGLE_MAP = {
    "left_elbow":  (L_SHO, L_ELB, L_WRI),
    "right_elbow": (R_SHO, R_ELB, R_WRI),
    "left_knee":   (L_HIP, L_KNE, L_ANK),
    "right_knee":  (R_HIP, R_KNE, R_ANK),
    "left_armpit": (L_ELB, L_SHO, L_HIP),
    "right_armpit":(R_ELB, R_SHO, R_HIP),
    "left_hip":    (L_SHO, L_HIP, L_KNE),
    "right_hip":   (R_SHO, R_HIP, R_KNE),
    "left_neck":   (L_ELB, L_SHO, R_SHO),
    "right_neck":  (R_ELB, R_SHO, L_SHO),
}

# ------------------ ScoreEngine ------------------
class ScoreEngine:
    """
    读取参考 JSON → 实时接收每 5 帧角度 → 输出 (score, label)
    """
    def __init__(self,
                 angle_json_path: str = "angle_data.json",
                 sample_rate: float = 15,
                 sim_threshold: float = 0.70,
                 window: float = 0.01):
        self.sample_rate = sample_rate
        self.threshold   = sim_threshold
        self.window      = window

        # 解析参考角度
        ref: List[Dict] = json.loads(Path(angle_json_path).read_text(encoding="utf‑8"))
        self._ref_frames: List[Dict] = []
        for item in ref:
            ts = _timestamp_to_sec(item["timestamp"])
            for ang_dict in item["angles"]:
                self._ref_frames.append({"t": ts, "angles": ang_dict})
        self._ref_frames.sort(key=lambda x: x["t"])

        self._t0 = None  # 开始时间

    # ---------- 公共接口 ----------
    def start(self) -> None:
        self._t0 = time.time()

    def kpts_to_angles(self, kpts: np.ndarray) -> Dict[str, float]:
        """(17,2) → 10 关节角度字典（单位: 弧度）"""
        out = {}
        for name, (i, j, k) in ANGLE_MAP.items():
            out[name] = _angle(kpts[i], kpts[j], kpts[k])
        return out

    def update_batch(self, batch_angles: List[Dict[str, float]]) -> Tuple[float, str]:
        """
        输入：长度=5 的角度字典列表  
        返回：(最佳相似度, 标签)；若未命中时间窗则 (None, "")
        """
        if self._t0 is None:
            self.start()
        t_now = time.time() - self._t0

        # 1. 找到时间窗内候选参考帧
        candidates = [rf for rf in self._ref_frames
                      if abs(rf["t"] - t_now) <= self.window]
        if not candidates:
            return None, ""

        # 2. 逐个比较取最大相似度
        best_sim, best_idx = 0.0, -1
        for idx, cand in enumerate(candidates):
            ref_vec = np.array([cand["angles"][k] for k in sorted(ANGLE_MAP)])
            if np.isnan(ref_vec).any():
                continue
            norm_ref = np.linalg.norm(ref_vec)
            for ang in batch_angles:
                cam_vec = np.array([ang[k] for k in sorted(ANGLE_MAP)])
                if np.isnan(cam_vec).any():
                    continue
                sim = float(np.dot(ref_vec, cam_vec) /
                            (norm_ref * np.linalg.norm(cam_vec)))
                if sim > best_sim:
                    best_sim, best_idx = sim, self._ref_frames.index(cand)

        # 3. 阈值判定 + 去重
        if best_sim < self.threshold:
            return 0.0, "MISS"
        # 移除已匹配参考帧
        self._ref_frames.pop(best_idx)
        return best_sim, score_to_label(best_sim)

# ------------------ 标签函数 ------------------
def score_to_label(score: float) -> str:
    for thresh, label in LABELS:
        if score >= thresh:
            return label
    return "MISS"