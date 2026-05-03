"""
PhysFormer rPPG 实时推理：多人脸轨迹 + 160 帧缓冲，估计 HR（BPM）。
权重路径见 config.PHYSFORMER_WEIGHTS；模型代码位于 physformer/PhysFormer。
"""
from __future__ import annotations

import os
import sys
import threading
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

# --- import PhysFormer from vendor tree ---
_FYP_ROOT = Path(__file__).resolve().parent
_PF_REPO = _FYP_ROOT / "physformer" / "PhysFormer"
if _PF_REPO.is_dir():
    sys.path.insert(0, str(_PF_REPO))

CLIP_FRAMES = 160
FACE_SIZE = 128


def _iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh
    inter_w = max(0, min(ax2, bx2) - max(ax, bx))
    inter_h = max(0, min(ay2, by2) - max(ay, by))
    inter = inter_w * inter_h
    if inter <= 0:
        return 0.0
    ua = aw * ah + bw * bh - inter
    return float(inter / ua) if ua > 0 else 0.0


def _hr_bpm_from_rppg(sig: np.ndarray, fps: float) -> float:
    x = sig.astype(np.float64) - np.mean(sig)
    spec = np.abs(np.fft.rfft(x))
    freqs = np.fft.rfftfreq(sig.shape[0], d=1.0 / fps)
    band = (freqs >= 0.65) & (freqs <= 3.5)
    if not np.any(band):
        return float("nan")
    k = int(np.argmax(spec[band]))
    bpm = float(freqs[band][k] * 60.0)
    return float(np.clip(bpm, 40.0, 200.0))


class PhysFormerEngine:
    """
    维护多条人脸轨迹；每条累积 CLIP_FRAMES 帧 128x128 BGR 后推理一次，
    滑动步长 SLIDE 帧以免过久才更新。
    """

    IOU_MATCH = 0.25
    MAX_MISS = 12
    SLIDE = 80

    def __init__(
        self,
        weights_path: Path,
        fps: float,
        device: Optional[torch.device] = None,
    ):
        self.fps = float(fps)
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self._weights_path = Path(weights_path)
        self.model = None
        self._load_error: Optional[str] = None
        self._load_model()

        self._tracks: Dict[int, Dict[str, Any]] = {}
        self._next_id = 0
        self._hr_history: Dict[int, deque] = {}
        self._lock = threading.Lock()

    def _load_model(self) -> None:
        if not self._weights_path.is_file():
            self._load_error = "missing file: %s" % self._weights_path
            return
        try:
            from model import ViT_ST_ST_Compact3_TDC_gra_sharp
        except ImportError as e:
            self._load_error = "PhysFormer import failed: %s" % e
            return

        try:
            m = ViT_ST_ST_Compact3_TDC_gra_sharp(
                image_size=(160, 128, 128),
                patches=(4, 4, 4),
                dim=96,
                ff_dim=144,
                num_heads=4,
                num_layers=12,
                dropout_rate=0.1,
                theta=0.7,
            )
            try:
                state = torch.load(
                    str(self._weights_path),
                    map_location=self.device,
                    weights_only=False,
                )
            except TypeError:
                state = torch.load(str(self._weights_path), map_location=self.device)
            m.load_state_dict(state, strict=True)
            self.model = m.to(self.device).eval()
            self._load_error = None
        except Exception as e:
            self.model = None
            self._load_error = str(e)

    @property
    def ok(self) -> bool:
        return self.model is not None

    @property
    def error(self) -> Optional[str]:
        return self._load_error

    def _smooth_hr(self, tid: int, hr: float) -> float:
        h = self._hr_history.setdefault(tid, deque(maxlen=5))
        if not np.isnan(hr) and 40 <= hr <= 200:
            h.append(hr)
        if not h:
            return float("nan")
        return float(np.median(h))

    def _match(self, boxes: List[Tuple[int, int, int, int]]) -> List[int]:
        """返回每个当前框对应的 track_id（可能新建）。"""
        if not self._tracks:
            ids = []
            for _ in boxes:
                tid = self._next_id
                self._next_id += 1
                ids.append(tid)
            return ids

        track_ids = list(self._tracks.keys())
        tboxes = [self._tracks[t]["bbox"] for t in track_ids]
        used_t = set()
        assigned: List[Optional[int]] = [None] * len(boxes)

        # greedy: for each current box, best unused track
        for i, box in enumerate(boxes):
            best_j = -1
            best_iou = self.IOU_MATCH
            for j, tb in enumerate(tboxes):
                if j in used_t:
                    continue
                v = _iou(box, tb)
                if v > best_iou:
                    best_iou = v
                    best_j = j
            if best_j >= 0:
                used_t.add(best_j)
                assigned[i] = track_ids[best_j]

        for i, box in enumerate(boxes):
            if assigned[i] is None:
                tid = self._next_id
                self._next_id += 1
                assigned[i] = tid
        return [int(a) for a in assigned]

    def step(
        self, frame_bgr: np.ndarray, face_boxes: List[Tuple[int, int, int, int]]
    ) -> List[Dict[str, Any]]:
        """
        face_boxes: 与画人脸顺序一致 (x,y,w,h)。
        返回每条脸的展示数据（供 OSD / API）。
        """
        with self._lock:
            return self._step_impl(frame_bgr, face_boxes)

    def _step_impl(
        self, frame_bgr: np.ndarray, face_boxes: List[Tuple[int, int, int, int]]
    ) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        if not self.ok:
            return out
        if not face_boxes:
            self._decay_tracks(set())
            return out

        H, W = frame_bgr.shape[:2]
        matched_ids = self._match(face_boxes)
        active: set = set()

        for tid, box in zip(matched_ids, face_boxes):
            active.add(tid)
            x, y, bw, bh = box
            x2, y2 = min(W, x + bw), min(H, y + bh)
            x, y = max(0, x), max(0, y)
            crop = frame_bgr[y:y2, x:x2]
            if crop.size == 0:
                continue
            face128 = cv2.resize(crop, (FACE_SIZE, FACE_SIZE), interpolation=cv2.INTER_CUBIC)

            tr = self._tracks.setdefault(
                tid,
                {
                    "bbox": box,
                    "buf": deque(maxlen=CLIP_FRAMES),
                    "hr": None,
                    "miss": 0,
                    "progress": 0,
                },
            )
            tr["bbox"] = box
            tr["miss"] = 0
            tr["buf"].append(face128.astype(np.uint8))
            tr["progress"] = len(tr["buf"])

            if len(tr["buf"]) == CLIP_FRAMES:
                hr_raw = self._infer_clip(tr["buf"])
                tr["hr"] = self._smooth_hr(tid, hr_raw)
                # slide window
                for _ in range(min(self.SLIDE, len(tr["buf"]))):
                    if tr["buf"]:
                        tr["buf"].popleft()
                tr["progress"] = len(tr["buf"])

            entry = {
                "track_id": tid,
                "hr_bpm": tr["hr"],
                "buffer_progress": tr["progress"],
                "buffer_need": CLIP_FRAMES,
            }
            out.append(entry)

        self._decay_tracks(active)
        return out

    def _decay_tracks(self, active: set) -> None:
        dead = []
        for tid in list(self._tracks.keys()):
            if tid not in active:
                self._tracks[tid]["miss"] = self._tracks[tid].get("miss", 0) + 1
                if self._tracks[tid]["miss"] > self.MAX_MISS:
                    dead.append(tid)
        for tid in dead:
            self._tracks.pop(tid, None)
            self._hr_history.pop(tid, None)

    @torch.no_grad()
    def _infer_clip(self, buf: deque) -> float:
        assert len(buf) == CLIP_FRAMES
        # (T, H, W, C) float [-1,1]
        vid = np.stack(list(buf), axis=0).astype(np.float32)
        vid = (vid - 127.5) / 128.0
        vid = np.transpose(vid, (3, 0, 1, 2))
        x = torch.from_numpy(vid).float().unsqueeze(0).to(self.device)
        gra_sharp = 2.0
        rppg, _, _, _ = self.model(x, gra_sharp)
        rppg = rppg.squeeze(0).cpu().numpy()
        rppg = (rppg - np.mean(rppg)) / (np.std(rppg) + 1e-8)
        return _hr_bpm_from_rppg(rppg, self.fps)

    def snapshot_tracks(self) -> Dict[str, Any]:
        return {
            "enabled": self.ok,
            "error": self.error,
            "weights": str(self._weights_path),
            "device": str(self.device),
            "faces": [],
        }


def resolve_weights_path(explicit: Optional[str]) -> Path:
    if explicit and Path(explicit).is_file():
        return Path(explicit).resolve()
    env = os.environ.get("PHYSFORMER_WEIGHTS", "").strip()
    if env and Path(env).is_file():
        return Path(env).resolve()
    candidates = [
        _FYP_ROOT / "weights" / "Physformer_UBFC_best.pkl",
        _FYP_ROOT / "physformer" / "PhysFormer" / "Physformer_UBFC_finetune" / "Physformer_UBFC_1_8.pkl",
        _FYP_ROOT / "weights" / "Physformer_VIPL_fold1.pkl",
    ]
    for c in candidates:
        if c.is_file():
            return c.resolve()
    return _FYP_ROOT / "weights" / "Physformer_UBFC_best.pkl"
