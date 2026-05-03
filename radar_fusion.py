"""
将摄像头多人脸（深度排序）与雷达多距离门通道对齐，用于融合展示。

算法层增强：
- 按缓冲进度与雷达质量加权融合 HR/RR；分歧较大时软化低置信侧权重。
- 可选：按 RealSense 测得的距离（米）将人物匹配到最相近的雷达距离门通道（启发式）。
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple


def _finite(x: Any) -> bool:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return False
    return not math.isnan(v) and not math.isinf(v)


def _visual_hr_weight(buffer_progress: Any, buffer_need: Any) -> float:
    """PhysFormer 缓冲未满时降低视觉心率置信度。"""
    try:
        need = float(buffer_need)
    except (TypeError, ValueError):
        need = 0.0
    if need <= 0:
        return 0.28
    try:
        p = float(buffer_progress or 0) / need
    except (TypeError, ValueError):
        p = 0.0
    p = max(0.0, min(1.0, p))
    return float(0.18 + 0.82 * (p**1.25))


def _radar_weight(q: Any, default: float = 0.38) -> float:
    if q is None:
        return default
    try:
        w = float(q)
    except (TypeError, ValueError):
        return default
    if math.isnan(w):
        return default
    return max(0.1, min(1.0, w))


def _visual_resp_weight() -> float:
    """绿通道呼吸无显式 SNR，给中等先验；融合时略低于雷达（胸位移更直接）。"""
    return 0.42


def _radar_hr_ok(x: Any) -> bool:
    if not _finite(x):
        return False
    v = float(x)
    return 44.0 <= v <= 195.0


def _radar_rr_ok(x: Any) -> bool:
    if not _finite(x):
        return False
    v = float(x)
    return 5.0 <= v <= 52.0


def _visual_hr_ok(x: Any) -> bool:
    if not _finite(x):
        return False
    v = float(x)
    return 44.0 <= v <= 195.0


def _visual_rr_ok(x: Any) -> bool:
    if not _finite(x):
        return False
    v = float(x)
    return 5.0 <= v <= 52.0


def _weighted_fuse_pair(
    va: Optional[float],
    wa: float,
    vb: Optional[float],
    wb: float,
    disagree_rel: float = 0.22,
) -> Tuple[Optional[float], str]:
    """双模态加权；相对分歧大时压低较小权重一侧。"""
    a_ok = va is not None and _finite(va) and va > 0
    b_ok = vb is not None and _finite(vb) and vb > 0
    if not a_ok and not b_ok:
        return None, "none"
    if a_ok and not b_ok:
        return float(va), "single_a"
    if b_ok and not a_ok:
        return float(vb), "single_b"
    assert va is not None and vb is not None
    fa, fb = float(va), float(vb)
    m = max(fa, fb, 1.0)
    rel = abs(fa - fb) / m
    wea, web = float(wa), float(wb)
    if rel > disagree_rel:
        if wea <= web:
            wea *= 0.45
        else:
            web *= 0.45
    s = wea + web
    if s < 1e-9:
        return 0.5 * (fa + fb), "mean_fallback"
    out = (wea * fa + web * fb) / s
    return float(out), "weighted"


def _greedy_depth_channel_assign(
    indexed_sorted: List[Tuple[Any, Optional[float], Dict[str, Any]]],
    ch_sorted: List[Dict[str, Any]],
    d_min: float,
    d_max: float,
) -> Tuple[List[Optional[Dict[str, Any]]], List[Optional[int]]]:
    """
    将场景深度 [d_min, d_max] 线性映射到通道下标 0..N-1，
    再按与首选通道 id 距离最小做贪心分配（解决多人抢同一通道）。
    """
    n_c = len(ch_sorted)
    used_j: set = set()
    assign_ch: List[Optional[Dict[str, Any]]] = []
    pref_ids: List[Optional[int]] = []
    for rank, (_oi, depth, _vf) in enumerate(indexed_sorted):
        pref: Optional[int] = None
        if n_c == 0:
            pref_ids.append(None)
            assign_ch.append(None)
            continue
        if depth is not None and _finite(depth) and d_max > d_min:
            t = (float(depth) - d_min) / (d_max - d_min)
            t = max(0.0, min(1.0, t))
            pref = int(round(t * max(0, n_c - 1)))
        else:
            pref = min(rank, n_c - 1)
        pref_ids.append(pref)
        best_j = None
        best_score = 1e18
        for j in range(n_c):
            if j in used_j:
                continue
            cid = int(ch_sorted[j].get("id", j))
            score = abs(cid - int(pref))
            if score < best_score:
                best_score = score
                best_j = j
        if best_j is None:
            assign_ch.append(None)
        else:
            used_j.add(best_j)
            assign_ch.append(ch_sorted[best_j])
    return assign_ch, pref_ids


def fuse_multiperson_vitals(
    vitals_faces: List[Dict[str, Any]],
    face_depths_m: List[Optional[float]],
    radar_payload: Optional[Dict[str, Any]],
    depth_scene_m: Optional[Tuple[float, float]] = None,
    use_depth_bin_match: bool = True,
) -> Dict[str, Any]:
    """
    vitals_faces: physformer_engine 输出的 faces（可含 depth_m / depth_zone 等）
    face_depths_m: 与脸顺序对齐的深度（米），None 表示未知
    radar_payload: Pi /api/radar 的 JSON（含 radar_channels 列表）
    depth_scene_m: (近界, 远界) 米，用于把深度映射到雷达通道下标
    use_depth_bin_match: True 时按深度→通道 id 贪心匹配；False 时维持名次配对
    """
    channels = []
    if radar_payload and isinstance(radar_payload, dict):
        channels = radar_payload.get("radar_channels") or radar_payload.get(
            "channels"
        )
        if not isinstance(channels, list):
            channels = []

    n = max(len(vitals_faces), len(face_depths_m))

    indexed = []
    for i in range(n):
        depth = face_depths_m[i] if i < len(face_depths_m) else None
        vf = vitals_faces[i] if i < len(vitals_faces) else {}
        indexed.append((i, depth, vf))
    indexed.sort(key=lambda x: (x[1] is None, x[1] if x[1] is not None else 1e9))

    ch_sorted = sorted(
        [c for c in channels if isinstance(c, dict)],
        key=lambda c: int(c.get("bin_start", 0)),
    )

    d_min, d_max = (0.35, 2.3)
    if depth_scene_m is not None and len(depth_scene_m) >= 2:
        d_min, d_max = float(depth_scene_m[0]), float(depth_scene_m[1])
    if d_max <= d_min:
        d_min, d_max = 0.35, 2.3

    if use_depth_bin_match and ch_sorted:
        ch_assign, pref_ids = _greedy_depth_channel_assign(
            indexed, ch_sorted, d_min, d_max
        )
        assignment_key = "depth_greedy_bin_match"
    else:
        ch_assign = [
            ch_sorted[i] if i < len(ch_sorted) else None for i in range(len(indexed))
        ]
        pref_ids = [None] * len(indexed)
        assignment_key = "depth_rank_to_bin_rank"

    people: List[Dict[str, Any]] = []
    for rank, (_old_i, depth, vf) in enumerate(indexed):
        ch = ch_assign[rank] if rank < len(ch_assign) else None
        row = {
            "slot": rank + 1,
            "track_id": vf.get("track_id"),
            "depth_m": vf.get("depth_m") if vf.get("depth_m") is not None else depth,
            "depth_zone": vf.get("depth_zone"),
            "depth_valid_ratio": vf.get("depth_valid_ratio"),
            "visual_hr_bpm": vf.get("hr_bpm"),
            "buffer_progress": vf.get("buffer_progress"),
            "buffer_need": vf.get("buffer_need"),
            "visual_resp_bpm": vf.get("resp_bpm"),
            "radar_hr_bpm": None,
            "radar_resp_bpm": None,
            "radar_bin_start": None,
            "radar_bin_end": None,
            "radar_channel_id": None,
            "radar_channel_pref_id": pref_ids[rank] if rank < len(pref_ids) else None,
            "radar_hq": None,
            "radar_bq": None,
            "fused_hr_bpm": None,
            "fused_resp_bpm": None,
            "fusion_hr_method": None,
            "fusion_rr_method": None,
        }
        if ch is not None:
            row["radar_hr_bpm"] = ch.get("heart_rate")
            row["radar_resp_bpm"] = ch.get("breathing_rate")
            row["radar_bin_start"] = ch.get("bin_start")
            row["radar_bin_end"] = ch.get("bin_end")
            row["radar_channel_id"] = ch.get("id")
            row["radar_hq"] = ch.get("heart_quality")
            row["radar_bq"] = ch.get("breathing_quality")

        vh = row["visual_hr_bpm"]
        vhr = row["visual_resp_bpm"]
        rh = row["radar_hr_bpm"]
        rr = row["radar_resp_bpm"]

        w_vis_hr = _visual_hr_weight(
            row.get("buffer_progress"), row.get("buffer_need")
        )
        w_rad_hr = _radar_weight(row.get("radar_hq"), 0.4)
        w_vis_rr = _visual_resp_weight()
        w_rad_rr = _radar_weight(row.get("radar_bq"), 0.45)

        vh_p = float(vh) if _visual_hr_ok(vh) else None
        rh_p = float(rh) if _radar_hr_ok(rh) else None
        vrr_p = float(vhr) if _visual_rr_ok(vhr) else None
        rrr_p = float(rr) if _radar_rr_ok(rr) else None

        fh, hm = _weighted_fuse_pair(vh_p, w_vis_hr, rh_p, w_rad_hr)
        fr, rm = _weighted_fuse_pair(vrr_p, w_vis_rr, rrr_p, w_rad_rr, disagree_rel=0.26)

        row["fused_hr_bpm"] = round(fh, 1) if fh is not None else None
        row["fused_resp_bpm"] = round(fr, 1) if fr is not None else None
        row["fusion_hr_method"] = hm
        row["fusion_rr_method"] = rm

        people.append(row)

    return {
        "people": people,
        "radar_channel_count": len(ch_sorted),
        "assignment": assignment_key,
        "fusion_model": "quality_weighted_disagreement_soften",
        "depth_scene_m": [d_min, d_max],
        "use_depth_bin_match": bool(use_depth_bin_match),
    }
