# score_engine.py
"""用于 Part 3 Task 2 的实时舞蹈评分引擎。

本模块独立于 danceapp.py 主程序，以避免主程序变得臃肿。
在 danceapp.py 中导入本模块，在每帧将参考动作和用户动作关键点传入
以获取一个实时分数及分类反馈（如“Perfect”、“Good”）。

评分指标：
-----------
1. **空间相似度** - 利用关键点之间的归一化欧氏距离衡量。
2. **动作趋势相似度** - 两帧之间的位移向量的余弦相似度。
3. **节拍对齐度** - 检测关键点加速度变化的峰值时刻，
   与音乐节拍的重合率。

最终评分为三项子评分的加权平均（范围 [0, 1]）。默认权重为 (0.4, 0.3, 0.3)，
可在初始化时自定义。

使用示例：
-----------
>>> from score_engine import ScoreEngine
>>> scorer = ScoreEngine(beat_audio_path="song.mp3")
>>> # 在主循环中：
>>> overall, details = scorer.update(ref_kpts, cam_kpts, timestamp)
>>> print(overall, details)
"""

# 以下为 Python 类型声明及所需库导入
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Dict

import numpy as np
from scipy.spatial.distance import cdist
from scipy.spatial.transform import Rotation as R
from scipy.signal import find_peaks

# 尝试导入 librosa（用于音乐节拍分析）
try:
    import librosa
except ImportError:
    librosa = None

# 类型别名：关键点为 (N, 2) 形状的 NumPy 数组，未检测点设为 NaN
Keypoints = np.ndarray

# 计算归一化欧氏距离（用于空间评分）
def _norm_d(n: Keypoints, r: Keypoints) -> float:
    def _scale(k: Keypoints) -> float:
        if k.shape[0] > 12 and not np.any(np.isnan(k[[5, 6]])):
            return np.linalg.norm(k[5] - k[6]) + 1e-5
        if k.shape[0] > 12 and not np.any(np.isnan(k[[11, 12]])):
            return np.linalg.norm(k[11] - k[12]) + 1e-5
        valid = k[~np.isnan(k).any(axis=1)]
        if valid.size == 0:
            return 1.0
        min_xy, max_xy = valid.min(axis=0), valid.max(axis=0)
        return np.linalg.norm(max_xy - min_xy) + 1e-5

    scale = 0.5 * (_scale(n) + _scale(r))
    dist = cdist(n, r, metric="euclidean").diagonal()
    valid = ~np.isnan(dist)
    if valid.sum() == 0:
        return 1.0
    return float(dist[valid].mean() / scale)

# 计算帧间位移向量（动作趋势）
def _motion_vec(k_prev: Keypoints, k_curr: Keypoints) -> np.ndarray:
    disp = k_curr - k_prev
    disp[np.isnan(disp)] = 0.0
    return disp.flatten()

# 用于保存三项子评分
@dataclass
class ScoreBreakdown:
    spatial: float
    motion: float
    beat: float

    def as_dict(self) -> Dict[str, float]:
        return {"spatial": self.spatial, "motion": self.motion, "beat": self.beat}

# 核心评分类
class ScoreEngine:
    def __init__(
        self,
        sample_rate: float = 30.0,
        beat_audio_path: str | None = None,
        weights: Tuple[float, float, float] = (0.6, 0.4, 0.0),
        beat_tolerance: float = 0.15,
        accel_window: int = 5,
        use_beat: bool = False, 
    ) -> None:
        self.fps = sample_rate
        self.use_beat = use_beat 
        self.weights = np.asarray(weights, dtype=float) / sum(weights)
        self.beat_tol = beat_tolerance
        self.accel_window = accel_window

        self._ref_prev: Keypoints | None = None
        self._cam_prev: Keypoints | None = None
        self._ref_vel_buf: List[np.ndarray] = []
        self._cam_vel_buf: List[np.ndarray] = []
        if self.use_beat and beat_audio_path and librosa is not None:  ### MODIFIED
            self.beat_times = self._extract_beats(beat_audio_path)
        else:
            self.beat_times = []

        self._ref_peak_times: List[float] = []
        self._cam_peak_times: List[float] = []

    # 主接口：输入关键点与时间戳，返回实时评分与详细分解
    def update(
        self,
        ref_kpts: Keypoints,
        cam_kpts: Keypoints,
        t: float,
    ) -> Tuple[float, ScoreBreakdown]:
        spatial_dist = _norm_d(ref_kpts, cam_kpts)
        spatial_score = np.exp(-spatial_dist ** 2)

        if self._ref_prev is not None and self._cam_prev is not None:
            ref_motion = _motion_vec(self._ref_prev, ref_kpts)
            cam_motion = _motion_vec(self._cam_prev, cam_kpts)
            num = np.dot(ref_motion, cam_motion)
            denom = (np.linalg.norm(ref_motion) * np.linalg.norm(cam_motion) + 1e-8)
            motion_score = (num / denom + 1) / 2
        else:
            motion_score = 0.5

        beat_score = self._update_beat_alignment(ref_kpts, cam_kpts, t)

        breakdown = ScoreBreakdown(spatial_score, motion_score, beat_score)
        overall = float((self.weights * np.array(list(breakdown.as_dict().values()))).sum())

        self._ref_prev = ref_kpts.copy()
        self._cam_prev = cam_kpts.copy()
        return overall, breakdown

    # 使用 librosa 提取音乐节拍
    def _extract_beats(self, path: str) -> List[float]:
        y, sr = librosa.load(path, mono=True)
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
        return librosa.frames_to_time(beats, sr=sr).tolist()

    # 更新节拍对齐率
    def _update_beat_alignment(
        self, ref_k: Keypoints, cam_k: Keypoints, t: float
    ) -> float:
        def _vel_buf(buf: List[np.ndarray], k_prev: Keypoints, k_curr: Keypoints):
            vel = _motion_vec(k_prev, k_curr) if k_prev is not None else np.zeros(k_curr.size)
            buf.append(vel)
            if len(buf) > self.accel_window:
                buf.pop(0)
            return buf

        self._ref_vel_buf = _vel_buf(self._ref_vel_buf, self._ref_prev, ref_k)
        self._cam_vel_buf = _vel_buf(self._cam_vel_buf, self._cam_prev, cam_k)

        if len(self._ref_vel_buf) < 3:
            return 0.5

        def _accel_mag(buf: List[np.ndarray]) -> float:
            acc = buf[-1] - buf[-2]
            return float(np.linalg.norm(acc))

        ref_acc = _accel_mag(self._ref_vel_buf)
        cam_acc = _accel_mag(self._cam_vel_buf)

        def _is_peak(a: float, history: List[float]) -> bool:
            if len(history) < 5:
                return False
            mu, sigma = np.mean(history), np.std(history)
            return a > mu + sigma

        if not hasattr(self, "_ref_acc_hist"):
            self._ref_acc_hist = []
            self._cam_acc_hist = []
        self._ref_acc_hist.append(ref_acc)
        self._cam_acc_hist.append(cam_acc)
        if _is_peak(ref_acc, self._ref_acc_hist):
            self._ref_peak_times.append(t)
        if _is_peak(cam_acc, self._cam_acc_hist):
            self._cam_peak_times.append(t)

        if len(self.beat_times) == 0 or len(self._cam_peak_times) == 0:
            return 0.5

        window = 5.0
        beats_win = [b for b in self.beat_times if t - window <= b <= t]
        peaks_win = [p for p in self._cam_peak_times if t - window <= p <= t]
        if len(beats_win) == 0:
            return 0.5
        aligned = sum(
            any(abs(p - b) <= self.beat_tol for b in beats_win) for p in peaks_win
        )
        return aligned / max(len(peaks_win), 1)

# 数值评分 → 文本标签（用于屏幕显示）
LABELS = [
    (0.90, "PERFECT"),
    (0.75, "SUPER"),
    (0.60, "GOOD"),
    (0.40, "OK"),
    (0.00, "MISS"),
]

def score_to_label(score: float) -> str:
    for thresh, label in LABELS:
        if score >= thresh:
            return label
    return "MISS"
