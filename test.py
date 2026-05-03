import pyrealsense2 as rs
import numpy as np
import cv2
from ultralytics import YOLO
from config import (
    WIDTH,
    HEIGHT,
    FPS,
    WINDOW_NAME,
    DEPTH_WINDOW_NAME,
    YOLO_MODEL,
    YOLO_CONFIDENCE,
    YOLO_IMGSZ,
    YOLO_DEVICE,
    MIN_BBOX_WIDTH,
    MIN_BBOX_HEIGHT,
    TRACK_IOU_THRESHOLD,
    TRACK_MIN_HITS,
    TRACK_MAX_MISSES,
    TRACK_DRAW_MAX_MISSES,
    DEPTH_VIS_ALPHA,
    DEPTH_VALID_MIN_M,
    DEPTH_VALID_MAX_M,
    DEPTH_MIN_VALID_PIXELS,
)
from realsense_utils import median_depth_meters


def compute_iou(box_a, box_b):
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b

    a_x2, a_y2 = ax + aw, ay + ah
    b_x2, b_y2 = bx + bw, by + bh

    inter_x1 = max(ax, bx)
    inter_y1 = max(ay, by)
    inter_x2 = min(a_x2, b_x2)
    inter_y2 = min(a_y2, b_y2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = aw * ah
    area_b = bw * bh
    union_area = area_a + area_b - inter_area
    if union_area <= 0:
        return 0.0
    return inter_area / union_area


def main():
    # 创建管道和配置对象
    pipeline = rs.pipeline()
    rs_config = rs.config()

    # 启用彩色流 + 深度流
    rs_config.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, FPS)
    rs_config.enable_stream(rs.stream.depth, WIDTH, HEIGHT, rs.format.z16, FPS)

    # 启动管道
    pipeline.start(rs_config)
    print("RealSense stream started. Press 'q' to quit.")

    # 将深度帧对齐到彩色帧，便于给检测框读距离
    align = rs.align(rs.stream.color)
    depth_scale = (
        pipeline.get_active_profile()
        .get_device()
        .first_depth_sensor()
        .get_depth_scale()
    )

    model = YOLO(YOLO_MODEL)
    print(f"YOLO model loaded: {YOLO_MODEL}")
    tracks = []
    next_track_id = 1

    try:
        while True:
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)

            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()
            if not color_frame or not depth_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())

            valid_detections = []
            yolo_result = model.predict(
                color_image,
                conf=YOLO_CONFIDENCE,
                classes=[0],  # person
                imgsz=YOLO_IMGSZ,
                device=YOLO_DEVICE,
                verbose=False,
            )[0]

            for b in yolo_result.boxes:
                x1, y1, x2, y2 = b.xyxy[0].tolist()
                x, y = int(x1), int(y1)
                w, h = int(x2 - x1), int(y2 - y1)
                if w < MIN_BBOX_WIDTH or h < MIN_BBOX_HEIGHT:
                    continue
                conf = float(b.conf[0].item()) if b.conf is not None else 0.0
                valid_detections.append((x, y, w, h, conf))

            matched_track_indices = set()
            matched_detection_indices = set()

            for det_idx, det in enumerate(valid_detections):
                x, y, w, h, conf = det
                best_iou = 0.0
                best_track_idx = -1
                for track_idx, track in enumerate(tracks):
                    if track_idx in matched_track_indices:
                        continue
                    iou = compute_iou((x, y, w, h), track["bbox"])
                    if iou > best_iou:
                        best_iou = iou
                        best_track_idx = track_idx

                if best_track_idx >= 0 and best_iou >= TRACK_IOU_THRESHOLD:
                    tracks[best_track_idx]["bbox"] = (x, y, w, h)
                    tracks[best_track_idx]["conf"] = conf
                    tracks[best_track_idx]["distance_m"] = median_depth_meters(
                        depth_image,
                        depth_scale,
                        x,
                        y,
                        w,
                        h,
                        WIDTH,
                        HEIGHT,
                        DEPTH_VALID_MIN_M,
                        DEPTH_VALID_MAX_M,
                        DEPTH_MIN_VALID_PIXELS,
                    )
                    tracks[best_track_idx]["hits"] += 1
                    tracks[best_track_idx]["misses"] = 0
                    matched_track_indices.add(best_track_idx)
                    matched_detection_indices.add(det_idx)

            for track_idx, track in enumerate(tracks):
                if track_idx not in matched_track_indices:
                    track["misses"] += 1

            tracks = [t for t in tracks if t["misses"] <= TRACK_MAX_MISSES]

            for det_idx, det in enumerate(valid_detections):
                if det_idx in matched_detection_indices:
                    continue
                x, y, w, h, conf = det
                tracks.append(
                    {
                        "id": next_track_id,
                        "bbox": (x, y, w, h),
                        "conf": conf,
                        "distance_m": median_depth_meters(
                            depth_image,
                            depth_scale,
                            x,
                            y,
                            w,
                            h,
                            WIDTH,
                            HEIGHT,
                            DEPTH_VALID_MIN_M,
                            DEPTH_VALID_MAX_M,
                            DEPTH_MIN_VALID_PIXELS,
                        ),
                        "hits": 1,
                        "misses": 0,
                    }
                )
                next_track_id += 1

            person_count = 0
            for track in tracks:
                if track["hits"] < TRACK_MIN_HITS:
                    continue
                # 允许短时丢帧继续显示，减少闪烁
                if track["misses"] > TRACK_DRAW_MAX_MISSES:
                    continue

                x, y, w, h = track["bbox"]
                cx = int(x + w / 2)
                cy = int(y + h / 2)
                person_count += 1

                dm = track["distance_m"]
                dist_str = f"{dm:.2f} m" if dm == dm else "depth n/a"

                cv2.rectangle(color_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                label = f"person {dist_str} conf:{track['conf']:.2f}"
                cv2.putText(
                    color_image,
                    label,
                    (x, max(y - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )
                cv2.circle(color_image, (cx, cy), 4, (255, 0, 0), -1)

            cv2.putText(
                color_image,
                f"persons: {person_count} raw:{len(valid_detections)} conf>={YOLO_CONFIDENCE}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
            )

            depth_colormap = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_image, alpha=DEPTH_VIS_ALPHA),
                cv2.COLORMAP_JET,
            )

            cv2.imshow(WINDOW_NAME, color_image)
            cv2.imshow(DEPTH_WINDOW_NAME, depth_colormap)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
            # 兼容点击窗口右上角关闭按钮退出
            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        print("Stream stopped.")


if __name__ == "__main__":
    main()