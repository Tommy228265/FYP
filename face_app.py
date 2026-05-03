import json
import math
import threading
import time
import urllib.error
import urllib.request
from collections import Counter, deque
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import pyrealsense2 as rs
from flask import Flask, Response, jsonify, render_template, request

from config import (
    RADAR_PI_BASE,
    WIDTH,
    HEIGHT,
    FPS,
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
}

PERSON_KEYS = [f"person{i}" for i in range(1, FACE_MAX_PROFILES + 1)]
PERSON_LABELS = {person: f"人物{i}" for i, person in enumerate(PERSON_KEYS, start=1)}

profiles = {person: None for person in PERSON_KEYS}
enroll_buffers = {person: [] for person in PERSON_KEYS}
profile_ages = {person: None for person in PERSON_KEYS}
pending_enroll_ages = {person: None for person in PERSON_KEYS}
age_prediction_histories = {}

pipeline = None
align = None
depth_scale = 1.0

_embedder = None
_age_estimator = None
_physformer: Optional[PhysFormerEngine] = None


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
        _embedder = FaceEmbedder(
            min_face_size=FACE_MTCNN_MIN_FACE_SIZE,
            pretrained=FACE_PRETRAINED,
        )
    return _embedder


def get_age_estimator() -> AgeEstimator:
    global _age_estimator
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
            best_name = PERSON_LABELS[person]

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
    }


def build_recognition_result(labels: List[str], scores: List[float], ages: List[dict]):
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
        "is_known": any(label != "未知面容" for label in labels),
        "updated_at": time.time(),
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


def draw_label(frame: np.ndarray, text: str, x: int, y: int, color: Tuple[int, int, int]):
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.58
    thickness = 2
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    top = max(y - th - 14, 2)
    left = max(x, 2)
    right = min(left + tw + 14, WIDTH - 2)
    cv2.rectangle(frame, (left, top), (right, top + th + 12), color, -1)
    cv2.putText(frame, text, (left + 7, top + th + 6), font, scale, (255, 255, 255), thickness)


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


def start_camera():
    global pipeline, align, depth_scale
    pipeline = rs.pipeline()
    rs_config = rs.config()
    rs_config.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, FPS)
    rs_config.enable_stream(rs.stream.depth, WIDTH, HEIGHT, rs.format.z16, FPS)
    pipeline.start(rs_config)
    align = rs.align(rs.stream.color)
    depth_scale = get_depth_scale_from_pipeline(pipeline)


def stop_camera():
    global pipeline
    if pipeline is not None:
        pipeline.stop()
        pipeline = None


def frame_generator():
    global last_enroll_time, status_text, mode, enroll_person, last_recognition, last_vitals, last_vitals_fusion, last_depth_jpeg
    embedder = get_embedder()
    age_estimator = get_age_estimator()
    while True:
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

        with state_lock:
            local_mode = mode
            local_person = enroll_person

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
                    depth_scale,
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
                            depth_scale,
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

        if local_mode == "enroll" and local_person in PERSON_KEYS and len(face_items) > 1:
            with state_lock:
                status_text = "录入时检测到多张人脸，请保持画面中只有当前录入者"

        for face_index, (emb, prob, blur, bbox) in enumerate(face_items):
            x, y, bw, bh = bbox
            x2, y2 = x + bw, y + bh

            dist_m = median_depth_meters(
                depth_image,
                depth_scale,
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
            quality_ok = prob_value >= FACE_MIN_MTCNN_PROB and (
                FACE_MIN_BLUR_VARIANCE <= 0 or blur >= FACE_MIN_BLUR_VARIANCE
            )

            now = time.time()
            label = f"Face p:{prob_value:.2f}"
            color = (200, 200, 0)

            if local_mode == "enroll" and local_person in PERSON_KEYS:
                if len(face_items) == 1 and quality_ok and (now - last_enroll_time) >= ENROLL_SAMPLE_INTERVAL_SEC:
                    with state_lock:
                        enroll_buffers[local_person].append(emb)
                        last_enroll_time = now
                        count = len(enroll_buffers[local_person])
                        person_label = PERSON_LABELS[local_person]
                        status_text = f"正在录入 {person_label}: {count}/{ENROLL_SAMPLES_TARGET}"
                        if count >= ENROLL_SAMPLES_TARGET:
                            template = build_profile_template(enroll_buffers[local_person])
                            duplicate_person, duplicate_score = find_duplicate_profile(
                                template,
                                exclude_person=local_person,
                            )
                            if duplicate_person is not None:
                                duplicate_label = PERSON_LABELS[duplicate_person]
                                status_text = (
                                    f"检测到重复面容：与{duplicate_label}相似度"
                                    f"{duplicate_score:.2f}，本次录入已取消"
                                )
                            else:
                                profiles[local_person] = template
                                profile_ages[local_person] = pending_enroll_ages.get(local_person)
                                save_profiles()
                                status_text = f"{person_label}录入完成"
                            enroll_buffers[local_person] = []
                            pending_enroll_ages[local_person] = None
                            mode = "idle"
                            enroll_person = None
                progress = len(enroll_buffers.get(local_person, []))
                label = f"Enroll {PERSON_LABELS[local_person]} {progress}/{ENROLL_SAMPLES_TARGET}"
                if len(face_items) > 1:
                    label += " Single face only"
                    color = (0, 165, 255)
                elif not quality_ok:
                    label += " Low quality"
                    color = (0, 165, 255)
                else:
                    color = (0, 255, 255)
            elif local_mode == "recognize":
                if quality_ok:
                    matched_person, name, score = classify_face_embedding(emb)
                    display_name = name if name != "Unknown" else "未知面容"
                    face_crop = frame[y:y2, x:x2]
                    age_result = age_estimator.estimate_from_bgr_crop(face_crop)
                    age_result = apply_age_calibration(age_result)
                    history_key = matched_person if matched_person is not None else f"unknown_{face_index}"
                    age_result = smooth_age_result(age_result, history_key)
                    age_result = add_age_evaluation(age_result, matched_person)
                    recognition_labels.append(display_name)
                    recognition_scores.append(score)
                    recognition_ages.append(age_result)
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
                    label = f"Low quality p:{prob_value:.2f} blur:{blur:.0f}"
                    color = (0, 165, 255)

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

        if local_mode == "recognize" and recognition_labels:
            with state_lock:
                last_recognition = build_recognition_result(
                    recognition_labels,
                    recognition_scores,
                    recognition_ages,
                )
                status_text = last_recognition["summary"]

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

        depth_vis = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_image, alpha=DEPTH_VIS_ALPHA),
            cv2.COLORMAP_TURBO,
        )
        for _emb, _pr, _bl, bbox in face_items:
            fx, fy, fw, fh = bbox
            cv2.rectangle(
                depth_vis,
                (int(fx), int(fy)),
                (int(fx + fw), int(fy + fh)),
                (0, 255, 100),
                2,
            )
        ok_depth, jpg_depth = cv2.imencode(".jpg", depth_vis)
        if ok_depth:
            with state_lock:
                last_depth_jpeg = jpg_depth.tobytes()

        ok, jpg = cv2.imencode(".jpg", frame)
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
        profile_items = [
            {
                "id": person,
                "label": PERSON_LABELS[person],
                "ready": profiles[person] is not None,
                "samples": len(enroll_buffers[person]),
                "actual_age": profile_ages[person],
            }
            for person in PERSON_KEYS
        ]
        return jsonify(
            {
                "mode": mode,
                "status_text": status_text,
                "max_profiles": FACE_MAX_PROFILES,
                "ready_count": sum(1 for profile in profiles.values() if profile is not None),
                "target_samples": ENROLL_SAMPLES_TARGET,
                "last_recognition": last_recognition,
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
            }
        )


@app.route("/api/enroll/start", methods=["POST"])
def api_enroll_start():
    global mode, enroll_person, status_text, last_enroll_time, last_recognition
    body = request.get_json(silent=True) or {}
    person = body.get("person")
    if person not in PERSON_KEYS:
        return jsonify({"ok": False, "error": f"person must be one of {', '.join(PERSON_KEYS)}"}), 400
    actual_age = body.get("actual_age")
    try:
        actual_age = int(actual_age)
    except (TypeError, ValueError):
        return jsonify({"ok": False, "error": "请先输入该人物的真实年龄"}), 400
    if actual_age < 0 or actual_age > 120:
        return jsonify({"ok": False, "error": "真实年龄应在 0 到 120 之间"}), 400

    with state_lock:
        mode = "enroll"
        enroll_person = person
        age_prediction_histories.clear()
        enroll_buffers[person] = []
        pending_enroll_ages[person] = actual_age
        last_enroll_time = 0.0
        last_recognition = reset_recognition("未识别")
        status_text = f"开始录入 {PERSON_LABELS[person]}（真实年龄 {actual_age} 岁），请正对摄像头并保持稳定"
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
        if enroll_person == person:
            enroll_person = None
            mode = "idle"
        last_recognition = reset_recognition("未识别")
        save_profiles()
        status_text = f"已删除 {PERSON_LABELS[person]} 的面容档案"
    return jsonify({"ok": True})


@app.route("/api/stop", methods=["POST"])
def api_stop():
    global mode, enroll_person, status_text, last_recognition
    with state_lock:
        mode = "idle"
        enroll_person = None
        age_prediction_histories.clear()
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


if __name__ == "__main__":
    load_profiles()
    start_camera()
    threading.Thread(target=radar_background_poll_loop, daemon=True).start()
    print("Facenet 人脸模型首次运行会下载权重，请稍候。")
    get_embedder()
    print(f"Age model: {get_age_estimator().status}")
    print("Web: http://127.0.0.1:5000  按 Ctrl+C 退出")
    try:
        app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
    finally:
        stop_camera()
