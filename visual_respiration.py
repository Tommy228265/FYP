"""
简易视觉呼吸辅助：人脸 ROI 绿色通道均值 → 0.1–0.5 Hz 峰值估计呼吸率（BPM 次/分）。
仅作辅助，与雷达呼吸对照；非医疗级。
"""
from __future__ import annotations

from collections import deque
from typing import Dict, Optional

import numpy as np


class VisualRespirationTracker:
    RESP_WIN = 150  # ~5 s @ 30fps
    FPS_DEFAULT = 30.0

    def __init__(self, fps: float = 30.0):
        self.fps = float(fps)
        self._green: Dict[int, deque] = {}
        self._last_rr: Dict[int, float] = {}
        self._ema_rr: Dict[int, float] = {}
        self._ema_alpha = 0.32

    def push(self, track_id: int, face_bgr: np.ndarray) -> None:
        if face_bgr is None or face_bgr.size == 0:
            return
        g = face_bgr[:, :, 1].astype(np.float64).mean()
        buf = self._green.setdefault(
            track_id, deque(maxlen=max(self.RESP_WIN * 2, 200))
        )
        buf.append(g)

    def estimate(self, track_id: int) -> Optional[float]:
        buf = self._green.get(track_id)
        if not buf or len(buf) < self.RESP_WIN:
            return self._last_rr.get(track_id)
        x = np.asarray(list(buf)[-self.RESP_WIN :], dtype=np.float64)
        x = x - np.mean(x)
        if np.std(x) < 1e-9:
            return self._last_rr.get(track_id)
        spec = np.abs(np.fft.rfft(x))
        freqs = np.fft.rfftfreq(len(x), d=1.0 / self.fps)
        band = (freqs >= 0.12) & (freqs <= 0.55)
        if not np.any(band):
            return self._last_rr.get(track_id)
        sub = spec[band]
        fk = freqs[band]
        k = int(np.argmax(sub))
        rr = float(fk[k] * 60.0)
        rr = float(np.clip(rr, 6.0, 45.0))
        prev = self._ema_rr.get(track_id)
        if prev is not None and prev > 0:
            rr = self._ema_alpha * rr + (1.0 - self._ema_alpha) * prev
        self._ema_rr[track_id] = rr
        self._last_rr[track_id] = rr
        return rr

    def forget(self, track_id: int) -> None:
        self._green.pop(track_id, None)
        self._last_rr.pop(track_id, None)
        self._ema_rr.pop(track_id, None)

    def prune_except(self, keep_ids: set) -> None:
        for tid in list(self._green.keys()):
            if tid not in keep_ids:
                self.forget(tid)
