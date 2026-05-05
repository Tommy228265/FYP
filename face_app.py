import copy
import html
import json
import math
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import Counter, deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pyrealsense2 as rs
from flask import Flask, Response, jsonify, render_template, request

from config import (
    RADAR_PI_BASE,
    USE_PI_CAMERA,
    WIDTH,
    HEIGHT,
    FPS,
    DISPLAY_NAME_MAX_LEN,
    FACE_PROFILE_META_FILE,
    DEPTH_VALID_MIN_M,
    DEPTH_VALID_MAX_M,
    DEPTH_MIN_VALID_PIXELS,
    DEPTH_VITAL_OPTIMAL_MIN_M,
    DEPTH_VITAL_OPTIMAL_MAX_M,
    DEPTH_VIS_ALPHA,
    FUSION_DEPTH_SCENE_MIN_M,
    FUSION_DEPTH_SCENE_MAX_M,
    FUSION_USE_DEPTH_BIN_MATCH,
    FACE_PROFILE_FILE,
    FACE_PROFILE_VERSION,
    FACE_MAX_PROFILES,
    ENROLL_SAMPLES_TARGET,
    ENROLL_SAMPLE_INTERVAL_SEC,
    FACE_RECOGNITION_THRESHOLD,
    FACE_DUPLICATE_THRESHOLD,
    FACE_MIN_MTCNN_PROB,
    FACE_MIN_BLUR_VARIANCE,
    FACE_MTCNN_MIN_FACE_SIZE,
    FACE_PRETRAINED,
    AGE_MODEL_FILE,
    AGE_CLASSES,
    AGE_INPUT_SIZE,
    AGE_CALIBRATION_SHIFT,
    AGE_SMOOTHING_WINDOW,
    PHYSFORMER_ENABLED,
    PHYSFORMER_WEIGHTS,
    VIDEO_MINIMAL_OVERLAY,
    VIDEO_JPEG_QUALITY,
    VIDEO_JPEG_DEPTH_QUALITY,
    VIDEO_ENCODE_DEPTH_STREAM,
    FACE_DEPTH_QUALITY_GATE,
    RECOGNITION_LOCK_WINDOW_SEC,
    RECOGNITION_LOCK_MAJORITY,
    RECOGNITION_LOCK_MIN_VOTES,
    AGE_LOCK_ON_CORRECT,
)
from age_estimator import AgeEstimator
from face_identity import FaceEmbedder
from physformer_engine import PhysFormerEngine, resolve_weights_path
from radar_fusion import fuse_multiperson_vitals
from realsense_utils import (
    depth_zone_label,
    face_depth_valid_ratio,
    get_depth_scale_from_pipeline,
    median_depth_meters,
)
from visual_respiration import VisualRespirationTracker


app = Flask(__name__)

state_lock = threading.Lock()
mode = "idle"  # idle | enroll | recognize
enroll_person = None
last_enroll_time = 0.0
status_text = "等待操作"
last_vitals = {
    "enabled": False,
    "error": None,
    "weights": "",
    "faces": [],
}

last_radar_cache: dict = {}
last_vitals_fusion: dict = {
    "people": [],
    "radar_channel_count": 0,
    "assignment": "depth_rank_to_bin_rank",
}

last_depth_jpeg: Optional[bytes] = None

_resp_tracker: Optional[VisualRespirationTracker] = None

last_recognition = {
    "label": "未识别",
    "summary": "未识别",
    "labels": [],
    "scores": [],
    "ages": [],
    "score": 0.0,
    "is_known": False,
    "updated_at": 0.0,
    "recognition_locked": False,
    "locked_since": 0.0,
}

RECOGNITION_VOTE_BAD_LABELS = frozenset({"低质量人脸", "未知面容"})

_recognition_vote_events: deque = deque()
_recognition_locked = False
_recognition_lock_snapshot: Optional[dict] = None
_frozen_age_by_hist_key: Dict[str, dict] = {}

PERSON_KEYS = [f"person{i}" for i in range(1, FACE_MAX_PROFILES + 1)]
PERSON_LABELS = {person: f"人物{i}" for i, person in enumerate(PERSON_KEYS, start=1)}

profile_meta: Dict[str, Dict[str, Any]] = {p: {"display_name": ""} for p in PERSON_KEYS}
pending_enroll_display_name: Optional[str] = None
last_live_preview: Dict[str, Any] = {"faces": [], "updated_at": 0.0}

profiles = {person: None for person in PERSON_KEYS}
enroll_buffers = {person: [] for person in PERSON_KEYS}
profile_ages = {person: None for person in PERSON_KEYS}
pending_enroll_ages = {person: None for person in PERSON_KEYS}
age_prediction_histories = {}

pipeline = None
align = None
depth_scale = 1.0

camera_backend = "realsense"
_remote_cap = None
_remote_depth_cap = None
_pi_depth_feed_ok = False

_embedder = None
_age_estimator = None
_physformer: Optional[PhysFormerEngine] = None
_model_init_lock = threading.Lock()


def get_resp_tracker() -> VisualRespirationTracker:
    global _resp_tracker
    if _resp_tracker is None:
        _resp_tracker = VisualRespirationTracker(float(FPS))
    return _resp_tracker


def get_physformer() -> Optional[PhysFormerEngine]:
    global _physformer
    if not PHYSFORMER_ENABLED:
        return None
    if _physformer is None:
        wp = resolve_weights_path(PHYSFORMER_WEIGHTS or None)
        _physformer = PhysFormerEngine(wp, float(FPS))
        if _physformer.ok:
            print(
                "[PhysFormer] 已加载",
                wp,
                "device=",
                _physformer.device,
            )
        else:
            print("[PhysFormer] 未启用:", _physformer.error)
    return _physformer


def _pack_vitals(
    pe: Optional[PhysFormerEngine], face_rows: List[dict]
) -> dict:
    if not PHYSFORMER_ENABLED:
        return {
            "enabled": False,
            "error": "已关闭（设置 PHYSFORMER_ENABLED=0）",
            "weights": "",
            "faces": [],
        }
    if pe is None:
        return {
            "enabled": False,
            "error": "未初始化",
            "weights": "",
            "faces": [],
        }
    wp = resolve_weights_path(PHYSFORMER_WEIGHTS or None)
    return {
        "enabled": pe.ok,
        "error": pe.error,
        "weights": str(wp),
        "faces": face_rows,
    }


def get_embedder() -> FaceEmbedder:
    global _embedder
    if _embedder is None:
        with _model_init_lock:
            if _embedder is None:
                _embedder = FaceEmbedder(
                    min_face_size=FACE_MTCNN_MIN_FACE_SIZE,
                    pretrained=FACE_PRETRAINED,
                )
    return _embedder


def get_age_estimator() -> AgeEstimator:
    global _age_estimator
    if _age_estimator is None:
        with _model_init_lock:
            if _age_estimator is None:
                _age_estimator = AgeEstimator(
                    model_path=AGE_MODEL_FILE,
                    classes=AGE_CLASSES,
                    input_size=AGE_INPUT_SIZE,
                )
    return _age_estimator


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
    return float(np.dot(a, b) / denom)


def build_profile_template(embeddings: List[np.ndarray]) -> np.ndarray:
    stack = np.stack(embeddings, axis=0)
    template = np.mean(stack, axis=0)
    template /= np.linalg.norm(template) + 1e-8
    return template.astype(np.float32)


def find_duplicate_profile(template: np.ndarray, exclude_person: Optional[str] = None):
    best_person = None
    best_score = -1.0
    for person, profile in profiles.items():
        if person == exclude_person or profile is None:
            continue
        score = cosine_similarity(template, profile)
        if score > best_score:
            best_person = person
            best_score = score
    if best_person is not None and best_score >= FACE_DUPLICATE_THRESHOLD:
        return best_person, best_score
    return None, best_score


def age_to_class_index(age: int) -> int:
    if age <= 12:
        return 0
    if age <= 19:
        return 1
    if age <= 35:
        return 2
    if age <= 55:
        return 3
    return 4


def age_class_for_actual_age(age: Optional[int]):
    if age is None:
        return None
    index = age_to_class_index(age)
    return AGE_CLASSES[index] if index < len(AGE_CLASSES) else None


def age_result_from_class_index(class_index: int, confidence: float, source: dict):
    item = AGE_CLASSES[class_index]
    result = {
        "label": item.get("label", f"年龄段{class_index}"),
        "label_en": item.get("label_en", f"Age {class_index}"),
        "range": item.get("range", "-"),
        "class_index": class_index,
        "class_id": item.get("id", f"age_{class_index}"),
        "confidence": confidence,
    }
    for key in ("raw_label", "raw_range", "raw_class_index", "calibration_shift", "smoothed"):
        if key in source:
            result[key] = source[key]
    return result


def apply_age_calibration(age_result: dict):
    raw_index = int(age_result.get("class_index", -1))
    if raw_index < 0 or raw_index >= len(AGE_CLASSES):
        return age_result

    shifted_index = min(raw_index + AGE_CALIBRATION_SHIFT, len(AGE_CLASSES) - 1)
    calibrated = age_result_from_class_index(
        shifted_index,
        float(age_result.get("confidence", 0.0)),
        {
            "raw_label": age_result.get("label"),
            "raw_range": age_result.get("range"),
            "raw_class_index": raw_index,
            "calibration_shift": AGE_CALIBRATION_SHIFT,
        },
    )
    calibrated["calibrated"] = shifted_index != raw_index
    return calibrated


def smooth_age_result(age_result: dict, history_key: str):
    class_index = int(age_result.get("class_index", -1))
    if class_index < 0 or class_index >= len(AGE_CLASSES):
        return age_result

    history = age_prediction_histories.setdefault(
        history_key,
        deque(maxlen=AGE_SMOOTHING_WINDOW),
    )
    history.append(
        {
            "class_index": class_index,
            "confidence": float(age_result.get("confidence", 0.0)),
            "source": age_result,
        }
    )

    counts = Counter(item["class_index"] for item in history)
    best_count = max(counts.values())
    candidates = {idx for idx, count in counts.items() if count == best_count}
    # 平票时选择最近一次出现的类别，避免结果长期卡在旧状态。
    winning_index = next(
        item["class_index"]
        for item in reversed(history)
        if item["class_index"] in candidates
    )
    confidences = [
        item["confidence"]
        for item in history
        if item["class_index"] == winning_index
    ]
    smoothed = age_result_from_class_index(
        winning_index,
        float(sum(confidences) / len(confidences)) if confidences else 0.0,
        age_result,
    )
    smoothed["smoothed"] = True
    smoothed["smoothing_window"] = len(history)
    return smoothed


def add_age_evaluation(age_result: dict, matched_person: Optional[str]):
    if matched_person is None or matched_person not in profile_ages:
        age_result["evaluation"] = "不可比较"
        age_result["is_correct"] = None
        return age_result

    actual_age = profile_ages.get(matched_person)
    actual_class = age_class_for_actual_age(actual_age)
    if actual_age is None or actual_class is None:
        age_result["evaluation"] = "未填写真实年龄"
        age_result["is_correct"] = None
        return age_result

    predicted_range = age_result.get("range")
    actual_range = actual_class.get("range")
    is_correct = predicted_range == actual_range
    age_result["actual_age"] = actual_age
    age_result["actual_range"] = actual_range
    age_result["actual_label"] = actual_class.get("label")
    age_result["is_correct"] = is_correct
    age_result["evaluation"] = "预测正确" if is_correct else "预测错误"
    return age_result


def label_for_person(person: Optional[str]) -> str:
    if person is None:
        return "未知面容"
    meta = profile_meta.get(person) or {}
    dn = (meta.get("display_name") or "").strip()
    if dn:
        return dn
    if profiles.get(person) is not None:
        return "未命名"
    try:
        idx = PERSON_KEYS.index(person) + 1
        return f"槽位{idx}"
    except ValueError:
        return str(person)


def classify_face_embedding(emb: np.ndarray):
    ready_profiles = [(person, profile) for person, profile in profiles.items() if profile is not None]
    if not ready_profiles:
        return None, "Unknown", 0.0

    best_person = None
    best_name = "Unknown"
    best_score = -1.0

    for person, profile in ready_profiles:
        score = cosine_similarity(emb, profile)
        if score > best_score:
            best_person = person
            best_score = score
            best_name = label_for_person(person)

    if best_score < FACE_RECOGNITION_THRESHOLD:
        return None, "Unknown", best_score
    return best_person, best_name, best_score


def reset_recognition(label: str = "未识别"):
    return {
        "label": label,
        "summary": label,
        "labels": [],
        "scores": [],
        "ages": [],
        "score": 0.0,
        "is_known": False,
        "updated_at": 0.0,
        "recognition_locked": False,
        "locked_since": 0.0,
    }


def _clear_recognition_stability_locks():
    """停止识别、重新开识别、录入等时清空投票与冻结状态。"""
    global _recognition_locked, _recognition_lock_snapshot, _frozen_age_by_hist_key
    _recognition_vote_events.clear()
    _recognition_locked = False
    _recognition_lock_snapshot = None
    _frozen_age_by_hist_key.clear()


def _depth_zone_ok_for_face_quality(dist_m: Any) -> bool:
    """有有效深度读数时要求 optimal；无读数时不挡（Pi 纯 RGB 等）。"""
    if not FACE_DEPTH_QUALITY_GATE:
        return True
    if dist_m is None:
        return True
    try:
        fx = float(dist_m)
    except (TypeError, ValueError):
        return True
    if not np.isfinite(fx):
        return True
    return (
        depth_zone_label(
            fx,
            DEPTH_VITAL_OPTIMAL_MIN_M,
            DEPTH_VITAL_OPTIMAL_MAX_M,
        )
        == "optimal"
    )


def _recognition_vote_tuple(labels: List[str]) -> Optional[tuple]:
    if not labels or any(l in RECOGNITION_VOTE_BAD_LABELS for l in labels):
        return None
    return tuple(labels)


def _maybe_advance_recognition_lock(
    now: float,
    labels: List[str],
    scores: List[float],
    ages: List[dict],
) -> None:
    """在识别模式下根据滑动窗口多数票决定是否锁定当前识别结果。"""
    global _recognition_locked, _recognition_lock_snapshot
    if _recognition_locked or RECOGNITION_LOCK_WINDOW_SEC <= 0:
        return
    key = _recognition_vote_tuple(labels)
    if key is None:
        return
    _recognition_vote_events.append((now, key))
    win = RECOGNITION_LOCK_WINDOW_SEC
    while _recognition_vote_events and now - _recognition_vote_events[0][0] > win:
        _recognition_vote_events.popleft()
    window = [e for e in _recognition_vote_events if now - e[0] <= win]
    if len(window) < RECOGNITION_LOCK_MIN_VOTES:
        return
    top_key, top_n = Counter(e[1] for e in window).most_common(1)[0]
    if top_n / float(len(window)) < RECOGNITION_LOCK_MAJORITY:
        return
    if top_key != key:
        return
    ts = time.time()
    built = build_recognition_result(labels, scores, ages)
    built["recognition_locked"] = True
    built["locked_since"] = ts
    built["updated_at"] = ts
    _recognition_locked = True
    _recognition_lock_snapshot = built


def build_recognition_result(
    labels: List[str],
    scores: List[float],
    ages: List[dict],
    recognition_locked: bool = False,
    locked_since: float = 0.0,
):
    if not labels:
        return reset_recognition("未识别")

    parts = []
    for index, label in enumerate(labels):
        age = ages[index] if index < len(ages) else {}
        age_text = age.get("label", "年龄未知")
        age_range = age.get("range", "-")
        evaluation = age.get("evaluation")
        evaluation_text = f"，{evaluation}" if evaluation else ""
        parts.append(f"{label}（预测{age_text} {age_range}{evaluation_text}）")
    summary = "从左到右依次是：" + "，".join(parts)
    return {
        "label": labels[0],
        "summary": summary,
        "labels": labels,
        "scores": [float(score) for score in scores],
        "ages": ages,
        "score": float(max(scores)) if scores else 0.0,
        "is_known": any(l not in RECOGNITION_VOTE_BAD_LABELS for l in labels),
        "updated_at": time.time(),
        "recognition_locked": bool(recognition_locked),
        "locked_since": float(locked_since or 0.0),
    }


def save_profiles():
    data = {"version": np.int32(FACE_PROFILE_VERSION)}
    for person, profile in profiles.items():
        if profile is not None:
            data[person] = profile
            if profile_ages[person] is not None:
                data[f"{person}_age"] = np.int32(profile_ages[person])
    if len(data) > 1:
        np.savez(FACE_PROFILE_FILE, **data)
        return

    path = Path(FACE_PROFILE_FILE)
    if path.exists():
        path.unlink()


def _sanitize_display_name(raw: Any) -> str:
    if raw is None:
        return "未命名"
    s = str(raw).strip()
    if not s:
        return "未命名"
    return s[:DISPLAY_NAME_MAX_LEN]


def load_profile_meta():
    path = Path(FACE_PROFILE_META_FILE)
    if not path.is_file():
        return
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return
        for p in PERSON_KEYS:
            if p in data and isinstance(data[p], dict):
                dn = data[p].get("display_name") or ""
                profile_meta[p]["display_name"] = str(dn)[:DISPLAY_NAME_MAX_LEN]
    except Exception as exc:
        print("[WARN] load_profile_meta:", exc)


def save_profile_meta():
    path = Path(FACE_PROFILE_META_FILE)
    try:
        out = {p: dict(profile_meta[p]) for p in PERSON_KEYS}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
    except Exception as exc:
        print("[WARN] save_profile_meta:", exc)


def _age_dict_for_api(age_result: Optional[dict]) -> Optional[dict]:
    if not age_result:
        return None
    return {
        "label": age_result.get("label"),
        "label_en": age_result.get("label_en"),
        "range": age_result.get("range"),
        "confidence": float(age_result.get("confidence", 0) or 0),
        "class_index": int(age_result.get("class_index", -1)),
        "evaluation": age_result.get("evaluation"),
        "smoothed": age_result.get("smoothed"),
    }


def load_profiles():
    path = Path(FACE_PROFILE_FILE)
    if not path.exists():
        return
    loaded = np.load(path, allow_pickle=True)
    ver = int(np.asarray(loaded["version"]).reshape(-1)[0]) if "version" in loaded else 0
    if ver < FACE_PROFILE_VERSION:
        return
    for person in PERSON_KEYS:
        if person in loaded:
            profiles[person] = loaded[person].astype(np.float32)
        age_key = f"{person}_age"
        if age_key in loaded:
            profile_ages[person] = int(np.asarray(loaded[age_key]).reshape(-1)[0])


def draw_status_panel(frame: np.ndarray, local_mode: str, text: str, ready_count: int):
    panel_h = 74
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (WIDTH, panel_h), (20, 24, 34), -1)
    cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)

    mode_color = (0, 255, 255)
    if local_mode == "recognize":
        mode_color = (0, 220, 0)
    elif local_mode == "enroll":
        mode_color = (0, 180, 255)

    cv2.putText(frame, f"Mode: {local_mode}", (14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.62, mode_color, 2)
    cv2.putText(
        frame,
        f"Profiles: {ready_count}/{FACE_MAX_PROFILES}",
        (14, 57),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (230, 230, 230),
        2,
    )
    cv2.putText(frame, text[:48], (210, 43), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)


def _resolve_cjk_font_path() -> Optional[str]:
    env = (os.environ.get("FYP_OPENCV_FONT") or "").strip()
    if env and os.path.isfile(env):
        return env
    for p in (
        r"C:\Windows\Fonts\msyh.ttc",
        r"C:\Windows\Fonts\msyhbd.ttc",
        r"C:\Windows\Fonts\simhei.ttf",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    ):
        if os.path.isfile(p):
            return p
    return None


_pil_label_font = None  # cache: ImageFont or False


def _get_pil_label_font():
    global _pil_label_font
    if _pil_label_font is False:
        return None
    if _pil_label_font is not None:
        return _pil_label_font
    try:
        from PIL import ImageFont
    except ImportError:
        _pil_label_font = False
        return None
    path = _resolve_cjk_font_path()
    if not path:
        _pil_label_font = False
        return None
    try:
        _pil_label_font = ImageFont.truetype(path, 18)
        return _pil_label_font
    except Exception:
        _pil_label_font = False
        return None


def draw_label(frame: np.ndarray, text: str, x: int, y: int, color: Tuple[int, int, int]):
    """绘制人脸旁标签；中文/日文等非 ASCII 需系统字体或设置 FYP_OPENCV_FONT。"""
    pil_font = _get_pil_label_font()
    if pil_font is not None:
        try:
            from PIL import ImageDraw
            from PIL import Image as PILImage

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_im = PILImage.fromarray(rgb)
            draw = ImageDraw.Draw(pil_im)
            bbox = draw.textbbox((0, 0), text, font=pil_font)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]
            top = max(y - th - 14, 2)
            left = max(x, 2)
            right = min(left + tw + 14, WIDTH - 2)
            fill_rgb = (int(color[2]), int(color[1]), int(color[0]))
            draw.rectangle([left, top, right, top + th + 12], fill=fill_rgb)
            draw.text((left + 7, top + 4), text, font=pil_font, fill=(255, 255, 255))
            frame[:, :, :] = cv2.cvtColor(np.asarray(pil_im), cv2.COLOR_RGB2BGR)
            return
        except Exception:
            pass

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.58
    thickness = 2
    safe = "".join(c if ord(c) < 128 else "?" for c in text)
    (tw, th), _ = cv2.getTextSize(safe, font, scale, thickness)
    top = max(y - th - 14, 2)
    left = max(x, 2)
    right = min(left + tw + 14, WIDTH - 2)
    cv2.rectangle(frame, (left, top), (right, top + th + 12), color, -1)
    cv2.putText(frame, safe, (left + 7, top + th + 6), font, scale, (255, 255, 255), thickness)


def draw_recognition_banner(frame: np.ndarray, labels: List[str], scores: List[float], ages: List[dict]):
    if not labels:
        return

    display_labels = []
    for index, label in enumerate(labels):
        face_number = "".join(ch for ch in label if ch.isdigit())
        face_text = f"Face {face_number}" if face_number else "Unknown"
        age = ages[index] if index < len(ages) else {}
        age_text = age.get("label_en", "Age N/A")
        display_labels.append(f"{face_text} {age_text}")

    banner_text = "Left to right: " + ", ".join(display_labels)
    detail_text = "Similarities: " + ", ".join(f"{score:.2f}" for score in scores)
    color = (28, 140, 40) if any(label != "未知面容" for label in labels) else (32, 32, 220)
    overlay = frame.copy()
    top = HEIGHT - 86

    cv2.rectangle(overlay, (0, top), (WIDTH, HEIGHT), color, -1)
    cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, frame)
    cv2.putText(frame, banner_text[:62], (24, top + 38), cv2.FONT_HERSHEY_SIMPLEX, 0.82, (255, 255, 255), 3)
    cv2.putText(frame, detail_text, (26, top + 67), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (245, 245, 245), 2)


def _make_pi_missing_frame_bgr() -> np.ndarray:
    """上位机拉不到有效帧时给出可读提示（便于区分「黑屏」与「算法未跑」）。"""
    img = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    img[:] = (36, 36, 42)
    lines = [
        "No video from Raspberry Pi",
        "Open http://<Pi>:5000/camera/rgb in browser",
        "Fix USB camera on Pi (see Pi terminal: ls /dev/video*)",
        "Or set USE_PI_CAMERA=0 and use Intel RealSense on PC",
    ]
    y = 36
    for line in lines:
        cv2.putText(
            img,
            line[:78],
            (16, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42 if WIDTH >= 640 else 0.35,
            (210, 210, 220),
            1,
            cv2.LINE_AA,
        )
        y += int(26 * (HEIGHT / 480.0))
    return img


def _make_pi_depth_placeholder_bgr() -> np.ndarray:
    """上位机拉不到树莓派深度 MJPEG 时的占位（仍与 RGB 并排展示）。"""
    img = np.full((HEIGHT, WIDTH, 3), 32, dtype=np.uint8)
    cv2.putText(
        img,
        "No Pi depth MJPEG — open <Pi>:5000/camera/depth",
        (14, 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (200, 200, 210),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        img,
        "Needs RealSense depth+color on Pi; USB cam has no depth",
        (14, 64),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (140, 140, 155),
        1,
        cv2.LINE_AA,
    )
    return img


def _pi_depth_frame_to_bgr_display(dfr: np.ndarray) -> np.ndarray:
    """
    解码树莓派深度 MJPEG：可能是单通道灰度；须转为 BGR 再画彩色人脸框。
    伪彩极暗时用 CLAHE 提亮整幅（仅用于显示）。
    """
    if dfr.ndim == 2:
        out = cv2.cvtColor(dfr, cv2.COLOR_GRAY2BGR)
    elif dfr.shape[2] == 4:
        out = cv2.cvtColor(dfr, cv2.COLOR_BGRA2BGR)
    else:
        out = dfr
    if float(np.mean(out)) < 22:
        lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        out = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return out


def _ensure_pi_depth_capture():
    """树莓派 /camera/depth MJPEG（深度伪彩）；与 RGB 同源对齐时可画人脸框。"""
    global _remote_depth_cap
    base = (RADAR_PI_BASE or "").strip().rstrip("/")
    if not base:
        return None
    url = base + "/camera/depth"
    if _remote_depth_cap is not None and _remote_depth_cap.isOpened():
        return _remote_depth_cap
    if _remote_depth_cap is not None:
        try:
            _remote_depth_cap.release()
        except Exception:
            pass
        _remote_depth_cap = None

    caps_to_try: List[int] = []
    ffmpeg_cap = getattr(cv2, "CAP_FFMPEG", None)
    if ffmpeg_cap is not None:
        caps_to_try.append(int(ffmpeg_cap))
    any_cap = getattr(cv2, "CAP_ANY", None)
    if any_cap is not None:
        caps_to_try.append(int(any_cap))
    caps_to_try.append(0)
    _seen: set = set()
    _uniq: List[int] = []
    for x in caps_to_try:
        if x not in _seen:
            _seen.add(x)
            _uniq.append(x)
    caps_to_try = _uniq

    last_err = None
    cap = None
    for api in caps_to_try:
        try:
            c = cv2.VideoCapture(url, api)
            if c.isOpened():
                try:
                    c.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                except Exception:
                    pass
                cap = c
                print("[INFO] 树莓派深度 MJPEG 已打开 (%s) backend=%s" % (url, api))
                break
            try:
                c.release()
            except Exception:
                pass
        except Exception as e:
            last_err = e
    if cap is None:
        cap = cv2.VideoCapture(url)
    if cap is not None and cap.isOpened():
        for _ in range(8):
            cap.read()
    _remote_depth_cap = cap
    if not cap.isOpened():
        print(
            "[WARN] 无法打开树莓派深度流: %s （%s）"
            % (url, last_err if last_err else "仍将显示占位深度面板")
        )
    return _remote_depth_cap


def _ensure_pi_video_capture():
    """延迟打开树莓派 MJPEG；Windows 下 HTTP 流优先 FFmpeg 后端。"""
    global _remote_cap
    base = (RADAR_PI_BASE or "").strip().rstrip("/")
    if not base:
        return None
    url = base + "/camera/rgb"
    if _remote_cap is not None and _remote_cap.isOpened():
        return _remote_cap
    if _remote_cap is not None:
        try:
            _remote_cap.release()
        except Exception:
            pass
        _remote_cap = None

    caps_to_try: List[int] = []
    ffmpeg_cap = getattr(cv2, "CAP_FFMPEG", None)
    if ffmpeg_cap is not None:
        caps_to_try.append(int(ffmpeg_cap))
    any_cap = getattr(cv2, "CAP_ANY", None)
    if any_cap is not None:
        caps_to_try.append(int(any_cap))
    caps_to_try.append(0)
    _uniq: List[int] = []
    _seen: set = set()
    for x in caps_to_try:
        if x not in _seen:
            _seen.add(x)
            _uniq.append(x)
    caps_to_try = _uniq

    last_err = None
    cap = None
    for api in caps_to_try:
        try:
            c = cv2.VideoCapture(url, api)
            if c.isOpened():
                try:
                    c.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                except Exception:
                    pass
                cap = c
                print("[INFO] 树莓派 MJPEG 已打开 (%s) backend=%s" % (url, api))
                break
            try:
                c.release()
            except Exception:
                pass
        except Exception as e:
            last_err = e
    if cap is None:
        cap = cv2.VideoCapture(url)
    _remote_cap = cap
    if not cap.isOpened():
        print(
            "[ERROR] 无法打开树莓派 MJPEG: %s （%s）"
            % (url, last_err if last_err else "仍可用浏览器访问该 URL 排查")
        )
    return _remote_cap


def start_camera():
    global pipeline, align, depth_scale, camera_backend, _remote_cap, _remote_depth_cap
    if USE_PI_CAMERA:
        base = (RADAR_PI_BASE or "").strip().rstrip("/")
        if not base:
            raise SystemExit(
                "USE_PI_CAMERA=1 需要设置环境变量 RADAR_PI_BASE（树莓派服务地址，如 http://10.245.232.43:5000）"
            )
        camera_backend = "pi_http"
        if _remote_cap is not None:
            try:
                _remote_cap.release()
            except Exception:
                pass
            _remote_cap = None
        if _remote_depth_cap is not None:
            try:
                _remote_depth_cap.release()
            except Exception:
                pass
            _remote_depth_cap = None
        pipeline = None
        align = None
        print(
            "[INFO] 树莓派视频将在首帧时连接（避免阻塞启动）；%s/camera/rgb 与 %s/camera/depth"
            % (base.rstrip("/"), base.rstrip("/"))
        )
        return

    camera_backend = "realsense"
    if _remote_cap is not None:
        _remote_cap.release()
        _remote_cap = None
    pipeline = rs.pipeline()
    rs_config = rs.config()
    rs_config.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, FPS)
    rs_config.enable_stream(rs.stream.depth, WIDTH, HEIGHT, rs.format.z16, FPS)
    pipeline.start(rs_config)
    align = rs.align(rs.stream.color)
    depth_scale = get_depth_scale_from_pipeline(pipeline)


def stop_camera():
    global pipeline, camera_backend, _remote_cap, _remote_depth_cap
    if camera_backend == "pi_http":
        if _remote_cap is not None:
            _remote_cap.release()
            _remote_cap = None
        if _remote_depth_cap is not None:
            try:
                _remote_depth_cap.release()
            except Exception:
                pass
            _remote_depth_cap = None
        camera_backend = "realsense"
        pipeline = None
        return
    if pipeline is not None:
        pipeline.stop()
        pipeline = None


def frame_generator():
    global last_enroll_time, status_text, mode, enroll_person, last_recognition, last_vitals, last_vitals_fusion, last_depth_jpeg
    global last_live_preview, pending_enroll_display_name, _pi_depth_feed_ok
    global _recognition_locked, _recognition_lock_snapshot
    embedder = get_embedder()
    age_estimator = get_age_estimator()
    pi_read_fail = 0
    _jpg_rgb = [int(cv2.IMWRITE_JPEG_QUALITY), int(VIDEO_JPEG_QUALITY)]
    _jpg_depth = [int(cv2.IMWRITE_JPEG_QUALITY), int(VIDEO_JPEG_DEPTH_QUALITY)]
    _green = (0, 255, 0)
    while True:
        if camera_backend == "pi_http":
            cap = _ensure_pi_video_capture()
            if cap is None or not cap.isOpened():
                time.sleep(0.08)
                frame = _make_pi_missing_frame_bgr()
                pi_read_fail = 0
            else:
                ok, frame = cap.read()
                if (
                    not ok
                    or frame is None
                    or (isinstance(frame, np.ndarray) and frame.size == 0)
                ):
                    pi_read_fail += 1
                    if pi_read_fail >= 45:
                        frame = _make_pi_missing_frame_bgr()
                        pi_read_fail = 0
                    else:
                        time.sleep(0.02)
                        continue
                else:
                    pi_read_fail = 0
                    if frame.shape[1] != WIDTH or frame.shape[0] != HEIGHT:
                        frame = cv2.resize(frame, (WIDTH, HEIGHT))
            depth_image = np.zeros((HEIGHT, WIDTH), dtype=np.uint16)
            depth_scale_local = 0.001
        else:
            if pipeline is None:
                time.sleep(0.05)
                continue

            frames = pipeline.wait_for_frames()
            frames = align.process(frames)
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            if not color_frame or not depth_frame:
                continue

            frame = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())
            depth_scale_local = depth_scale

        with state_lock:
            local_mode = mode
            local_person = enroll_person
            local_pending_name = pending_enroll_display_name

        face_items = embedder.embed_all_from_bgr(frame)
        pe = get_physformer()
        vitals_list: List[dict] = []
        if pe is not None:
            vitals_list = pe.step(frame, [t[3] for t in face_items])
        depths_for_fusion: List[Optional[float]] = []
        if pe is not None and vitals_list:
            rt = get_resp_tracker()
            keep_ids = set()
            for face_index, (_, _, _, bbox) in enumerate(face_items):
                x, y, bw, bh = bbox
                x2, y2 = x + bw, y + bh
                dist_m = median_depth_meters(
                    depth_image,
                    depth_scale_local,
                    x,
                    y,
                    bw,
                    bh,
                    WIDTH,
                    HEIGHT,
                    DEPTH_VALID_MIN_M,
                    DEPTH_VALID_MAX_M,
                    DEPTH_MIN_VALID_PIXELS,
                )
                dm: Optional[float] = None
                if dist_m is not None:
                    try:
                        fx = float(dist_m)
                        if np.isfinite(fx):
                            dm = fx
                    except (TypeError, ValueError):
                        pass
                depths_for_fusion.append(dm)
                if face_index < len(vitals_list):
                    tid = vitals_list[face_index].get("track_id")
                    vitals_list[face_index]["depth_m"] = (
                        round(dm, 3) if dm is not None else None
                    )
                    vitals_list[face_index]["depth_zone"] = depth_zone_label(
                        dm,
                        DEPTH_VITAL_OPTIMAL_MIN_M,
                        DEPTH_VITAL_OPTIMAL_MAX_M,
                    )
                    vitals_list[face_index]["depth_valid_ratio"] = round(
                        face_depth_valid_ratio(
                            depth_image,
                            depth_scale_local,
                            x,
                            y,
                            bw,
                            bh,
                            WIDTH,
                            HEIGHT,
                            DEPTH_VALID_MIN_M,
                            DEPTH_VALID_MAX_M,
                        ),
                        3,
                    )
                    if tid is not None:
                        keep_ids.add(tid)
                        y0, y1 = max(0, y), min(y2, HEIGHT)
                        x0, x1 = max(0, x), min(x2, WIDTH)
                        crop = frame[y0:y1, x0:x1]
                        rt.push(int(tid), crop)
                        rr = rt.estimate(int(tid))
                        if rr is not None:
                            vitals_list[face_index]["resp_bpm"] = round(rr, 1)
            rt.prune_except(keep_ids)
            with state_lock:
                fusion = fuse_multiperson_vitals(
                    vitals_list,
                    depths_for_fusion,
                    last_radar_cache,
                    depth_scene_m=(
                        FUSION_DEPTH_SCENE_MIN_M,
                        FUSION_DEPTH_SCENE_MAX_M,
                    ),
                    use_depth_bin_match=FUSION_USE_DEPTH_BIN_MATCH,
                )
                last_vitals = _pack_vitals(pe, vitals_list)
                last_vitals_fusion = fusion
        else:
            with state_lock:
                fusion = fuse_multiperson_vitals(
                    vitals_list or [],
                    [],
                    last_radar_cache,
                    depth_scene_m=(
                        FUSION_DEPTH_SCENE_MIN_M,
                        FUSION_DEPTH_SCENE_MAX_M,
                    ),
                    use_depth_bin_match=FUSION_USE_DEPTH_BIN_MATCH,
                )
                last_vitals = _pack_vitals(pe, vitals_list)
                last_vitals_fusion = fusion

        recognition_labels = []
        recognition_scores = []
        recognition_ages = []
        live_preview_list: List[dict] = []

        if local_mode != "recognize":
            if _recognition_locked or _recognition_vote_events:
                _clear_recognition_stability_locks()
        elif len(face_items) == 0:
            if _recognition_locked or _recognition_vote_events:
                _clear_recognition_stability_locks()
        elif _recognition_locked and _recognition_lock_snapshot is not None:
            if len(face_items) != len(_recognition_lock_snapshot.get("labels", [])):
                _clear_recognition_stability_locks()

        if local_mode == "enroll" and local_person in PERSON_KEYS and len(face_items) > 1:
            with state_lock:
                status_text = "录入时检测到多张人脸，请保持画面中只有当前录入者"

        for face_index, (emb, prob, blur, bbox) in enumerate(face_items):
            x, y, bw, bh = bbox
            x2, y2 = x + bw, y + bh

            dist_m = median_depth_meters(
                depth_image,
                depth_scale_local,
                x,
                y,
                bw,
                bh,
                WIDTH,
                HEIGHT,
                DEPTH_VALID_MIN_M,
                DEPTH_VALID_MAX_M,
                DEPTH_MIN_VALID_PIXELS,
            )

            if emb is None:
                if VIDEO_MINIMAL_OVERLAY:
                    cv2.rectangle(frame, (x, y), (x2, y2), _green, 2)
                else:
                    cv2.rectangle(frame, (x, y), (x2, y2), (128, 128, 128), 2)
                    cv2.putText(
                        frame,
                        "Face detected (align failed)",
                        (x, max(y - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (128, 128, 128),
                        2,
                    )
                continue

            prob_value = prob if prob is not None else 0.0
            depth_ok = _depth_zone_ok_for_face_quality(dist_m)
            quality_ok = prob_value >= FACE_MIN_MTCNN_PROB and (
                FACE_MIN_BLUR_VARIANCE <= 0 or blur >= FACE_MIN_BLUR_VARIANCE
            ) and depth_ok

            age_raw = None
            age_live = None
            if quality_ok:
                face_crop = frame[y:y2, x:x2]
                age_raw = age_estimator.estimate_from_bgr_crop(face_crop)
                age_raw = apply_age_calibration(age_raw)
                age_live = smooth_age_result(copy.deepcopy(age_raw), f"live_{face_index}")

            dm_api = None
            dz_api = None
            if dist_m is not None:
                try:
                    fx = float(dist_m)
                    if np.isfinite(fx):
                        dm_api = round(fx, 3)
                        dz_api = depth_zone_label(
                            fx,
                            DEPTH_VITAL_OPTIMAL_MIN_M,
                            DEPTH_VITAL_OPTIMAL_MAX_M,
                        )
                except (TypeError, ValueError):
                    pass

            preview_quality_ok = quality_ok
            preview_age_result: Optional[dict] = age_live if quality_ok else None

            now = time.time()

            if local_mode == "enroll" and local_person in PERSON_KEYS:
                if len(face_items) == 1 and quality_ok and (now - last_enroll_time) >= ENROLL_SAMPLE_INTERVAL_SEC:
                    with state_lock:
                        enroll_buffers[local_person].append(emb)
                        last_enroll_time = now
                        count = len(enroll_buffers[local_person])
                        nm = local_pending_name or "未命名"
                        status_text = f"正在录入「{nm}」: {count}/{ENROLL_SAMPLES_TARGET}"
                        if count >= ENROLL_SAMPLES_TARGET:
                            template = build_profile_template(enroll_buffers[local_person])
                            duplicate_person, duplicate_score = find_duplicate_profile(
                                template,
                                exclude_person=local_person,
                            )
                            if duplicate_person is not None:
                                duplicate_label = label_for_person(duplicate_person)
                                status_text = (
                                    f"检测到重复面容：与「{duplicate_label}」相似度"
                                    f"{duplicate_score:.2f}，本次录入已取消"
                                )
                                pending_enroll_display_name = None
                            else:
                                profiles[local_person] = template
                                profile_ages[local_person] = pending_enroll_ages.get(local_person)
                                profile_meta[local_person]["display_name"] = _sanitize_display_name(
                                    local_pending_name
                                )
                                save_profiles()
                                save_profile_meta()
                                finished = profile_meta[local_person]["display_name"]
                                status_text = f"「{finished}」录入完成"
                                pending_enroll_display_name = None
                            enroll_buffers[local_person] = []
                            pending_enroll_ages[local_person] = None
                            mode = "idle"
                            enroll_person = None
                            _clear_recognition_stability_locks()
                progress = len(enroll_buffers.get(local_person, []))
                nm = local_pending_name or "未命名"
                label = f"录入「{nm}」 {progress}/{ENROLL_SAMPLES_TARGET}"
                if len(face_items) > 1:
                    label += " · 仅单人"
                    color = (0, 165, 255)
                elif not quality_ok:
                    label += " · 请保持稳定"
                    color = (0, 165, 255)
                else:
                    color = (0, 255, 255)
            elif local_mode == "recognize":
                use_frozen = (
                    _recognition_locked
                    and _recognition_lock_snapshot is not None
                    and len(face_items) == len(_recognition_lock_snapshot.get("labels", []))
                    and face_index < len(_recognition_lock_snapshot["labels"])
                )
                if use_frozen:
                    sl = _recognition_lock_snapshot
                    display_name = str(sl["labels"][face_index])
                    score = (
                        float(sl["scores"][face_index])
                        if face_index < len(sl["scores"])
                        else 0.0
                    )
                    ag: dict = {}
                    if face_index < len(sl["ages"]):
                        ag = copy.deepcopy(sl["ages"][face_index])
                    recognition_labels.append(display_name)
                    recognition_scores.append(score)
                    recognition_ages.append(ag)
                    preview_quality_ok = True
                    preview_age_result = ag if ag else None
                    age_label = ag.get("label", "年龄未知")
                    age_range = ag.get("range", "-")
                    label = f"{display_name} {age_label}{age_range} sim:{score:.2f}"
                    color = (0, 255, 0) if display_name != "未知面容" else (0, 0, 255)
                elif quality_ok:
                    matched_person, name, score = classify_face_embedding(emb)
                    display_name = name if name != "Unknown" else "未知面容"
                    hist_key = matched_person if matched_person is not None else f"unknown_{face_index}"
                    if age_raw:
                        if AGE_LOCK_ON_CORRECT and hist_key in _frozen_age_by_hist_key:
                            age_result = copy.deepcopy(_frozen_age_by_hist_key[hist_key])
                        else:
                            age_result = smooth_age_result(copy.deepcopy(age_raw), hist_key)
                            age_result = add_age_evaluation(age_result, matched_person)
                            if AGE_LOCK_ON_CORRECT and age_result.get("is_correct") is True:
                                _frozen_age_by_hist_key[hist_key] = copy.deepcopy(age_result)
                    else:
                        age_result = {
                            "label": "无法估计",
                            "label_en": "N/A",
                            "range": "-",
                            "confidence": 0.0,
                        }
                    recognition_labels.append(display_name)
                    recognition_scores.append(score)
                    recognition_ages.append(age_result)
                    preview_age_result = age_result
                    age_label = age_result.get("label", "年龄未知")
                    age_range = age_result.get("range", "-")
                    label = f"{display_name} {age_label}{age_range} sim:{score:.2f}"
                    color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                else:
                    recognition_labels.append("低质量人脸")
                    recognition_scores.append(0.0)
                    recognition_ages.append(
                        {"label": "无法估计", "label_en": "N/A", "range": "-", "confidence": 0.0}
                    )
                    preview_quality_ok = False
                    preview_age_result = None
                    label = f"Low quality p:{prob_value:.2f} blur:{blur:.0f}"
                    color = (0, 165, 255)
            else:
                if quality_ok and age_live is not None:
                    age_label = age_live.get("label", "?")
                    age_range = age_live.get("range", "")
                    label = f"{age_label} · {age_range}"
                    color = (200, 200, 255)
                else:
                    label = f"Face p:{prob_value:.2f}"
                    if not quality_ok:
                        label += " · 需更清晰"
                    color = (200, 200, 0)

            pv = {
                "slot": face_index + 1,
                "quality_ok": preview_quality_ok,
                "age": _age_dict_for_api(preview_age_result) if preview_age_result else None,
                "depth_m": dm_api,
                "depth_zone": dz_api,
            }
            if not preview_quality_ok:
                pv["reason"] = "low_quality"
            live_preview_list.append(pv)

            if vitals_list and face_index < len(vitals_list):
                vd = vitals_list[face_index]
                hrv = vd.get("hr_bpm")
                prg = int(vd.get("buffer_progress", 0))
                need = int(vd.get("buffer_need", 160))
                if hrv is not None and isinstance(hrv, (int, float)) and not (
                    isinstance(hrv, float) and math.isnan(hrv)
                ):
                    label += f" | ~{float(hrv):.0f} bpm"
                else:
                    label += f" | 心率 {prg}/{need}"

            if VIDEO_MINIMAL_OVERLAY:
                cv2.rectangle(frame, (x, y), (x2, y2), _green, 2)
            else:
                cv2.rectangle(frame, (x, y), (x2, y2), color, 3)
                cv2.circle(frame, (x + bw // 2, y + bh // 2), 4, color, -1)
                if not np.isnan(dist_m):
                    cv2.putText(
                        frame,
                        f"{dist_m:.2f} m",
                        (x, min(y2 + 22, HEIGHT - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        color,
                        2,
                    )
                draw_label(frame, label, x, y, color)

        with state_lock:
            last_live_preview = {"faces": live_preview_list, "updated_at": time.time()}

        if local_mode == "recognize" and recognition_labels:
            now_rec = time.time()
            if not _recognition_locked:
                _maybe_advance_recognition_lock(
                    now_rec,
                    recognition_labels,
                    recognition_scores,
                    recognition_ages,
                )
            with state_lock:
                if _recognition_locked and _recognition_lock_snapshot is not None:
                    snap = copy.deepcopy(_recognition_lock_snapshot)
                    snap["updated_at"] = now_rec
                    last_recognition = snap
                    status_text = snap["summary"]
                else:
                    last_recognition = build_recognition_result(
                        recognition_labels,
                        recognition_scores,
                        recognition_ages,
                    )
                    status_text = last_recognition["summary"]

        if not VIDEO_MINIMAL_OVERLAY:
            with state_lock:
                draw_status_panel(
                    frame,
                    mode,
                    status_text,
                    sum(1 for profile in profiles.values() if profile is not None),
                )
                recent = time.time() - last_recognition["updated_at"] <= 3.0
                if mode == "recognize" and recent:
                    draw_recognition_banner(
                        frame,
                        last_recognition["labels"],
                        last_recognition["scores"],
                        last_recognition["ages"],
                    )

        if camera_backend == "pi_http":
            got_pi_depth = False
            d_cap = _ensure_pi_depth_capture()
            if d_cap is not None and d_cap.isOpened():
                okd, dfr = d_cap.read()
                if (
                    okd
                    and dfr is not None
                    and isinstance(dfr, np.ndarray)
                    and dfr.size > 0
                ):
                    got_pi_depth = True
                    dfr = _pi_depth_frame_to_bgr_display(dfr)
                    if dfr.shape[1] != WIDTH or dfr.shape[0] != HEIGHT:
                        dfr = cv2.resize(dfr, (WIDTH, HEIGHT))
                    depth_vis = dfr.copy()
                else:
                    depth_vis = _make_pi_depth_placeholder_bgr()
            else:
                depth_vis = _make_pi_depth_placeholder_bgr()
            _pi_depth_feed_ok = got_pi_depth
            dc = (0, 255, 0) if VIDEO_MINIMAL_OVERLAY else (0, 255, 100)
            for _emb, _pr, _bl, bbox in face_items:
                fx, fy, fw, fh = bbox
                cv2.rectangle(
                    depth_vis,
                    (int(fx), int(fy)),
                    (int(fx + fw), int(fy + fh)),
                    dc,
                    2,
                )
        else:
            depth_vis = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_image, alpha=DEPTH_VIS_ALPHA),
                cv2.COLORMAP_TURBO,
            )
            dc = (0, 255, 0) if VIDEO_MINIMAL_OVERLAY else (0, 255, 100)
            for _emb, _pr, _bl, bbox in face_items:
                fx, fy, fw, fh = bbox
                cv2.rectangle(
                    depth_vis,
                    (int(fx), int(fy)),
                    (int(fx + fw), int(fy + fh)),
                    dc,
                    2,
                )
        if VIDEO_ENCODE_DEPTH_STREAM:
            ok_depth, jpg_depth = cv2.imencode(".jpg", depth_vis, _jpg_depth)
            if ok_depth:
                with state_lock:
                    last_depth_jpeg = jpg_depth.tobytes()

        ok, jpg = cv2.imencode(".jpg", frame, _jpg_rgb)
        if not ok:
            continue
        chunk = jpg.tobytes()
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + chunk + b"\r\n"
        )


@app.route("/")
def index():
    return render_template("index.html")


def _readme_html() -> str:
    md_path = Path(__file__).resolve().parent / "README.md"
    raw = md_path.read_text(encoding="utf-8") if md_path.is_file() else "# 未找到 README.md"
    try:
        import markdown

        return markdown.markdown(
            raw,
            extensions=["tables", "fenced_code", "nl2br"],
        )
    except ImportError:
        return '<pre class="readme-fallback">%s</pre>' % html.escape(raw)


@app.route("/readme")
def readme_page():
    return render_template(
        "docs.html",
        title="项目说明与引用",
        body_html=_readme_html(),
    )


@app.route("/video_feed")
def video_feed():
    return Response(frame_generator(), mimetype="multipart/x-mixed-replace; boundary=frame")


def depth_stream_generator():
    while True:
        with state_lock:
            chunk = last_depth_jpeg
        if chunk:
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + chunk + b"\r\n"
            )
        time.sleep(1.0 / max(12.0, float(FPS)))


@app.route("/video_depth")
def video_depth():
    return Response(
        depth_stream_generator(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/api/status")
def api_status():
    with state_lock:
        profile_items = []
        for idx, person in enumerate(PERSON_KEYS):
            dn = (profile_meta[person].get("display_name") or "").strip()
            ready = profiles[person] is not None
            label = dn or ("未命名" if ready else f"槽位{idx + 1}")
            profile_items.append(
                {
                    "id": person,
                    "label": label,
                    "display_name": dn,
                    "ready": ready,
                    "samples": len(enroll_buffers[person]),
                    "actual_age": profile_ages[person],
                }
            )
        return jsonify(
            {
                "mode": mode,
                "status_text": status_text,
                "enroll_person": enroll_person,
                "enrolling_display_name": pending_enroll_display_name if mode == "enroll" else None,
                "max_profiles": FACE_MAX_PROFILES,
                "ready_count": sum(1 for profile in profiles.values() if profile is not None),
                "target_samples": ENROLL_SAMPLES_TARGET,
                "last_recognition": last_recognition,
                "live_preview": dict(last_live_preview),
                "age_model_status": get_age_estimator().status,
                "profiles": profile_items,
                "vitals": last_vitals,
                "vitals_fusion": last_vitals_fusion,
                "depth_camera": {
                    "optimal_range_m": [
                        DEPTH_VITAL_OPTIMAL_MIN_M,
                        DEPTH_VITAL_OPTIMAL_MAX_M,
                    ],
                    "fusion_scene_m": [
                        FUSION_DEPTH_SCENE_MIN_M,
                        FUSION_DEPTH_SCENE_MAX_M,
                    ],
                    "fusion_use_depth_bin_match": FUSION_USE_DEPTH_BIN_MATCH,
                },
                "sensor": {
                    "camera_backend": camera_backend,
                    "depth_available": True,
                    "use_pi_camera": USE_PI_CAMERA,
                    "pi_depth_live": (
                        _pi_depth_feed_ok if camera_backend == "pi_http" else None
                    ),
                },
            }
        )


@app.route("/api/enroll/start", methods=["POST"])
def api_enroll_start():
    global mode, enroll_person, status_text, last_enroll_time, last_recognition, pending_enroll_display_name
    body = request.get_json(silent=True) or {}
    person = body.get("person")
    if person not in PERSON_KEYS:
        return jsonify({"ok": False, "error": f"person must be one of {', '.join(PERSON_KEYS)}"}), 400
    raw_age = body.get("actual_age")
    actual_age = None
    if raw_age is not None and raw_age != "":
        try:
            actual_age = int(raw_age)
        except (TypeError, ValueError):
            return jsonify({"ok": False, "error": "真实年龄应为整数"}), 400
        if actual_age < 0 or actual_age > 120:
            return jsonify({"ok": False, "error": "真实年龄应在 0 到 120 之间"}), 400

    display_nm = _sanitize_display_name(body.get("display_name"))

    with state_lock:
        mode = "enroll"
        enroll_person = person
        pending_enroll_display_name = display_nm
        age_prediction_histories.clear()
        _clear_recognition_stability_locks()
        enroll_buffers[person] = []
        pending_enroll_ages[person] = actual_age
        last_enroll_time = 0.0
        last_recognition = reset_recognition("未识别")
        age_hint = f"，参考年龄 {actual_age} 岁" if actual_age is not None else "（未填写参考年龄，将无法比对预测准确度）"
        status_text = f"开始录入「{display_nm}」{age_hint}，请正对摄像头并保持稳定"
    return jsonify({"ok": True})


@app.route("/api/recognize/start", methods=["POST"])
def api_recognize_start():
    global mode, status_text, last_recognition
    with state_lock:
        ready_count = sum(1 for profile in profiles.values() if profile is not None)
        if ready_count == 0:
            return jsonify({"ok": False, "error": "请至少完成 1 个面容录入"}), 400
        mode = "recognize"
        age_prediction_histories.clear()
        _clear_recognition_stability_locks()
        last_recognition = reset_recognition("等待识别")
        status_text = f"识别模式运行中，当前已录入 {ready_count}/{FACE_MAX_PROFILES} 个面容"
    return jsonify({"ok": True})


@app.route("/api/profile/delete", methods=["POST"])
def api_profile_delete():
    global mode, enroll_person, status_text, last_recognition
    body = request.get_json(silent=True) or {}
    person = body.get("person")
    if person not in PERSON_KEYS:
        return jsonify({"ok": False, "error": f"person must be one of {', '.join(PERSON_KEYS)}"}), 400

    with state_lock:
        profiles[person] = None
        profile_ages[person] = None
        pending_enroll_ages[person] = None
        age_prediction_histories.pop(person, None)
        enroll_buffers[person] = []
        profile_meta[person] = {"display_name": ""}
        if enroll_person == person:
            enroll_person = None
            mode = "idle"
        _clear_recognition_stability_locks()
        last_recognition = reset_recognition("未识别")
        save_profiles()
        save_profile_meta()
        status_text = f"已删除槽位 {PERSON_KEYS.index(person) + 1} 的面容档案"
    return jsonify({"ok": True})


@app.route("/api/stop", methods=["POST"])
def api_stop():
    global mode, enroll_person, status_text, last_recognition
    with state_lock:
        mode = "idle"
        enroll_person = None
        age_prediction_histories.clear()
        _clear_recognition_stability_locks()
        last_recognition = reset_recognition("未识别")
        status_text = "已停止，等待操作"
    return jsonify({"ok": True})


def _radar_pi_url(path):
    base = (RADAR_PI_BASE or "").strip().rstrip("/")
    if not base:
        return None
    return base + path


def radar_background_poll_loop():
    """后台拉取树莓派 /api/radar，供多人融合与页面代理共用缓存。"""
    global last_radar_cache
    while True:
        time.sleep(0.45)
        if not (RADAR_PI_BASE or "").strip():
            continue
        data, err = _proxy_pi_json("GET", "/api/radar")
        if isinstance(data, dict) and err is None:
            with state_lock:
                last_radar_cache = data


def _proxy_pi_json(method, path, payload_bytes=None):
    url = _radar_pi_url(path)
    if not url:
        return None, "RADAR_PI_BASE 未配置"
    try:
        data = None
        if method != "GET":
            data = payload_bytes if payload_bytes is not None else b"{}"
        req = urllib.request.Request(
            url,
            data=data,
            method=method,
            headers={"Content-Type": "application/json", "User-Agent": "fyp-face-app"},
        )
        with urllib.request.urlopen(req, timeout=6.0) as resp:
            body = resp.read().decode("utf-8")
            return json.loads(body), None
    except urllib.error.HTTPError as e:
        try:
            detail = e.read().decode("utf-8")
        except Exception:
            detail = str(e)
        return None, "HTTP %s: %s" % (e.code, detail)
    except Exception as e:
        return None, str(e)


@app.route("/api/radar/poll")
def api_radar_poll():
    """浏览器轮询：转发树莓派 /api/radar 的生命体征数据。"""
    res, err = _proxy_pi_json("GET", "/api/radar")
    if err:
        return jsonify({"ok": False, "error": err, "radar": None})
    return jsonify({"ok": True, "radar": res})


@app.route("/api/radar/start", methods=["POST"])
def api_radar_start():
    data, err = _proxy_pi_json("POST", "/api/start-radar", b"{}")
    if err:
        return jsonify({"ok": False, "error": err})
    success = bool(data.get("success")) if isinstance(data, dict) else False
    msg = data.get("message", "") if isinstance(data, dict) else ""
    return jsonify({"ok": success, "message": msg})


@app.route("/api/radar/stop", methods=["POST"])
def api_radar_stop():
    data, err = _proxy_pi_json("POST", "/api/stop-radar", b"{}")
    if err:
        return jsonify({"ok": False, "error": err})
    success = bool(data.get("success")) if isinstance(data, dict) else True
    msg = data.get("message", "") if isinstance(data, dict) else ""
    return jsonify({"ok": success, "message": msg})


@app.route("/api/profile/display_name", methods=["POST"])
def api_profile_display_name():
    global status_text
    body = request.get_json(silent=True) or {}
    person = body.get("person")
    if person not in PERSON_KEYS:
        return jsonify({"ok": False, "error": f"person must be one of {', '.join(PERSON_KEYS)}"}), 400
    name = _sanitize_display_name(body.get("display_name"))
    with state_lock:
        profile_meta[person]["display_name"] = name
        save_profile_meta()
        status_text = f"已更新档案名称：「{name}」"
    return jsonify({"ok": True})


if __name__ == "__main__":
    load_profile_meta()
    load_profiles()
    if USE_PI_CAMERA:
        print(
            "[INFO] USE_PI_CAMERA=1：拉取树莓派 /camera/rgb 与 /camera/depth；"
            "深度量测融合仍以雷达通道配对为主（Pi 侧无原始深度矩阵上传）。"
        )
    else:
        print("[INFO] 使用本机 Intel RealSense（RGB + 深度）。")
    start_camera()
    threading.Thread(target=radar_background_poll_loop, daemon=True).start()

    def _preload_models():
        print("[INFO] Facenet 首次运行可能下载权重，请稍候…")
        get_embedder()
        print("[INFO] Age model: %s" % (get_age_estimator().status,))
        print("[OK] 人脸与年龄段模型已就绪")

    threading.Thread(target=_preload_models, daemon=True).start()
    print(
        "Web: http://127.0.0.1:5000  （/api/status 立即可用；识别模型后台加载中）按 Ctrl+C 退出"
    )
    try:
        app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
    finally:
        stop_camera()
