#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
树莓派端：毫米波雷达（与 jianjie.py 一致）+ USB 摄像头 MJPEG 推流；不做识别与生理推理。
浏览器与算法在上位机 face_app；摄像头由树莓派采集，上位机通过 USE_PI_CAMERA=1 拉流。
启动时可 POST 唤醒上位机 fyp_launcher 并尝试全屏打开 Chromium（与此前版本一致）。

环境变量（均有默认，直接: python3 shumeipai.py）：
  FYP_CAMERA_ENABLE=1（默认）  设为 0 可关闭摄像头推流（仅雷达）
  FYP_CAMERA_MODE=auto       auto：优先 Intel RealSense（同 jianjie.py）；失败再试 USB 摄像头
                             realsense：仅 RealSense   uvc：仅 OpenCV /dev/video*
  FYP_CAMERA_INDEX / FYP_CAMERA_WIDTH / FYP_CAMERA_HEIGHT / FYP_CAMERA_FPS
  FYP_CAMERA_DEVICE=          非空时直接打开该 V4L2 设备（如 /dev/video2），优先于索引
  FYP_CAMERA_PROBE=1          为 0 时禁用自动探测 /dev/video*（仅用 INDEX）
  FYP_REALSENSE_COLOR_ONLY=0  为 1 时跳过深度流（USB2/省电场景可试）
  FYP_PC_HOST=10.162.133.140
  FYP_UI_URL / FYP_LAUNCHER_URL   未单独设置时由 FYP_PC_HOST 自动拼出
  FYP_LAUNCHER_TOKEN   FYP_OPEN_KIOSK=1   FYP_AUTO_START=1   FYP_SERIAL=/dev/ttyUSB0
"""

import glob
import math
import os
import re
import sys
import serial
import subprocess
import threading
import time
import urllib.request
from collections import deque
from datetime import datetime

import cv2
import numpy as np
from flask import Flask, Response, jsonify
from scipy import signal

try:
    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
except Exception:
    pass

_DEFAULT_PC_HOST = "10.162.133.140"
_PC_HOST = (os.environ.get("FYP_PC_HOST") or "").strip() or _DEFAULT_PC_HOST
if _PC_HOST:
    if not (os.environ.get("FYP_UI_URL") or "").strip():
        os.environ["FYP_UI_URL"] = "http://%s:5000" % _PC_HOST
    if not (os.environ.get("FYP_LAUNCHER_URL") or "").strip():
        os.environ["FYP_LAUNCHER_URL"] = "http://%s:8787" % _PC_HOST

SERIAL_PORT = os.environ.get("FYP_SERIAL", "/dev/ttyUSB0")
HTTP_HOST = os.environ.get("FYP_BIND", "0.0.0.0")
HTTP_PORT = int(os.environ.get("FYP_PORT", "5000"))
FYP_UI_URL = os.environ.get("FYP_UI_URL", "").strip().rstrip("/")
FYP_LAUNCHER_URL = os.environ.get("FYP_LAUNCHER_URL", "").strip().rstrip("/")
FYP_LAUNCHER_TOKEN = os.environ.get("FYP_LAUNCHER_TOKEN", "").strip()

# USB 摄像头推流给上位机（默认开启；设 FYP_CAMERA_ENABLE=0 仅雷达）
FYP_CAMERA_ENABLE = os.environ.get("FYP_CAMERA_ENABLE", "1").strip().lower() not in (
    "0",
    "false",
    "no",
)
FYP_CAMERA_INDEX = int(os.environ.get("FYP_CAMERA_INDEX", "0"))
FYP_CAMERA_WIDTH = int(os.environ.get("FYP_CAMERA_WIDTH", "640"))
FYP_CAMERA_HEIGHT = int(os.environ.get("FYP_CAMERA_HEIGHT", "480"))
FYP_CAMERA_FPS = int(os.environ.get("FYP_CAMERA_FPS", "30"))
# auto | realsense | uvc — 与 jianjie.py 一致时 D435 用 realsense，不是 /dev/video0
FYP_CAMERA_MODE = (os.environ.get("FYP_CAMERA_MODE") or "auto").strip().lower()

_cam_lock = threading.Lock()
_latest_jpeg = None
_latest_depth_jpeg = None
_DEPTH_NA_JPEG_CACHE = None
# 旧版固定 alpha 易使远距离深度整幅发黑；现改为分位归一化伪彩
_cam_running = False
_cam_thread = None

app = Flask(__name__)


def _jpeg_placeholder_no_camera():
    """摄像头打不开时也推送有效 JPEG，便于上位机 OpenCV 拉到 MJPEG（否则会长期黑屏）。"""
    img = np.zeros((FYP_CAMERA_HEIGHT, FYP_CAMERA_WIDTH, 3), dtype=np.uint8)
    img[:] = (42, 42, 48)
    msg = "No camera — RealSense? use FYP_CAMERA_MODE=realsense  USB? ls /dev/video* idx=%s" % (
        FYP_CAMERA_INDEX,
    )
    cv2.putText(
        img,
        msg[:72],
        (12, FYP_CAMERA_HEIGHT // 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (220, 220, 230),
        1,
        cv2.LINE_AA,
    )
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 78])
    return buf.tobytes() if ok else None


def _depth_uint16_to_colormap_bgr(depth_uint16: np.ndarray) -> np.ndarray:
    """16 位深度 → TURBO 伪彩；按有效像素分位拉伸，避免画面一片黑。"""
    d = depth_uint16.astype(np.float32)
    valid = d > 0
    h, w = d.shape[:2]
    if not np.any(valid):
        return np.zeros((h, w, 3), dtype=np.uint8)
    lo = float(np.percentile(d[valid], 5))
    hi = float(np.percentile(d[valid], 95))
    if hi <= lo + 1e-3:
        hi = lo + 1.0
    scaled = (d - lo) / (hi - lo) * 255.0
    u8 = np.clip(scaled, 0, 255).astype(np.uint8)
    u8[~valid] = 0
    return cv2.applyColorMap(u8, cv2.COLORMAP_TURBO)


def _jpeg_depth_na():
    """无深度流时仍输出可读 JPEG（UVC / color-only RealSense）。"""
    global _DEPTH_NA_JPEG_CACHE
    if _DEPTH_NA_JPEG_CACHE is not None:
        return _DEPTH_NA_JPEG_CACHE
    img = np.zeros((FYP_CAMERA_HEIGHT, FYP_CAMERA_WIDTH, 3), dtype=np.uint8)
    img[:] = (26, 26, 32)
    cv2.putText(
        img,
        "Depth N/A (USB cam or color-only)",
        (12, FYP_CAMERA_HEIGHT // 2 - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (180, 180, 190),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        img,
        "RealSense depth: use depth+color on Pi",
        (12, FYP_CAMERA_HEIGHT // 2 + 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (130, 130, 145),
        1,
        cv2.LINE_AA,
    )
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 76])
    _DEPTH_NA_JPEG_CACHE = buf.tobytes() if ok else None
    return _DEPTH_NA_JPEG_CACHE


def _sorted_video_dev_paths():
    paths = glob.glob("/dev/video*")

    def _key(p):
        m = re.search(r"(\d+)$", p)
        return int(m.group(1)) if m else 0

    return sorted(paths, key=_key)


def _open_usb_camera():
    """Linux 上优先 V4L2；可选显式设备路径或自动探测第一个能出画的 /dev/video*。"""
    device = (os.environ.get("FYP_CAMERA_DEVICE") or "").strip()
    probe = os.environ.get("FYP_CAMERA_PROBE", "1").strip().lower() not in (
        "0",
        "false",
        "no",
    )
    if device:
        try:
            cap = (
                cv2.VideoCapture(device, cv2.CAP_V4L2)
                if sys.platform.startswith("linux")
                else cv2.VideoCapture(device)
            )
            if cap.isOpened():
                print("[CAMERA] UVC 使用 FYP_CAMERA_DEVICE=%s" % device)
                return cap
            cap.release()
        except Exception:
            pass
        print("[CAMERA] 无法打开 FYP_CAMERA_DEVICE=%s，将尝试索引/探测" % device)

    idx = FYP_CAMERA_INDEX
    if sys.platform.startswith("linux") and probe:
        for path in _sorted_video_dev_paths():
            try:
                cap = cv2.VideoCapture(path, cv2.CAP_V4L2)
                if not cap.isOpened():
                    cap.release()
                    continue
                ok, frame = cap.read()
                if ok and frame is not None and getattr(frame, "size", 0) > 0:
                    print("[CAMERA] UVC 自动探测使用 %s（可用 export FYP_CAMERA_DEVICE 固定）" % path)
                    return cap
                cap.release()
            except Exception:
                pass

    if sys.platform.startswith("linux"):
        try:
            cap_v4l = cv2.VideoCapture(idx, cv2.CAP_V4L2)
            if cap_v4l.isOpened():
                return cap_v4l
            cap_v4l.release()
        except Exception:
            pass
    return cv2.VideoCapture(idx)


def _uvc_camera_loop():
    """普通 USB 摄像头：OpenCV + V4L2（与 jianjie.py 中 Intel RealSense 方案不同）。"""
    global _latest_jpeg
    cap = _open_usb_camera()
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, FYP_CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FYP_CAMERA_HEIGHT)
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
    else:
        print(
            "[CAMERA] UVC 无法打开索引 %s（若实际是 Intel RealSense，请用默认 FYP_CAMERA_MODE=auto）"
            % FYP_CAMERA_INDEX
        )
    while _cam_running:
        if not cap.isOpened():
            pj = _jpeg_placeholder_no_camera()
            if pj:
                with _cam_lock:
                    _latest_jpeg = pj
            time.sleep(0.33)
            continue
        ok, frame = cap.read()
        if ok:
            if frame.shape[1] != FYP_CAMERA_WIDTH or frame.shape[0] != FYP_CAMERA_HEIGHT:
                frame = cv2.resize(frame, (FYP_CAMERA_WIDTH, FYP_CAMERA_HEIGHT))
            ok_j, buf = cv2.imencode(
                ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 82]
            )
            if ok_j:
                with _cam_lock:
                    _latest_jpeg = buf.tobytes()
        else:
            time.sleep(0.02)
    cap.release()


def _realsense_camera_loop():
    """
    与 jianjie.py 相同：pyrealsense2.pipeline + depth/color。
    D435 插在树莓派上时一般走这里，而不是 /dev/video0。
    """
    global _latest_jpeg
    try:
        import pyrealsense2 as rs
    except ImportError:
        print(
            "[CAMERA] 未找到 pyrealsense2。树莓派 ARM 上 PyPI 往往无预编译 wheel，"
            "请勿依赖 pip；请按 README「RealSense 与树莓派」从 Intel 文档编译安装。"
        )
        return False

    try:
        ctx = rs.context()
        n = len(list(ctx.query_devices()))
        print("[CAMERA] RealSense 枚举设备数: %s" % n)
        if n == 0:
            print(
                "[CAMERA] 提示：query_devices=0 时 pipeline 也会报 No device connected。"
                "请插紧 USB3、检查供电与线缆；摄像头是否在 PC 上占用？运行 rs-enumerate-devices 核对。"
            )
    except Exception as ex:
        print("[CAMERA] RealSense 枚举设备（可忽略）: %s" % ex)

    pipeline = rs.pipeline()
    w, h, fps = FYP_CAMERA_WIDTH, FYP_CAMERA_HEIGHT, FYP_CAMERA_FPS
    color_only_pref = (os.environ.get("FYP_REALSENSE_COLOR_ONLY") or "0").strip().lower() in (
        "1",
        "true",
        "yes",
    )

    def _stop_safe():
        try:
            pipeline.stop()
        except Exception:
            pass

    attempts = []
    if not color_only_pref:

        def _depth_color(c):
            c.enable_stream(rs.stream.depth, w, h, rs.format.z16, fps)
            c.enable_stream(rs.stream.color, w, h, rs.format.bgr8, fps)

        attempts.append(("depth+color %dx%d@%d" % (w, h, fps), _depth_color))

    def _color_wh(c):
        c.enable_stream(rs.stream.color, w, h, rs.format.bgr8, fps)

    attempts.append(("color-only %dx%d@%d" % (w, h, fps), _color_wh))
    attempts.append(("color-only 424x240@15（兼容 USB2/带宽不足）", lambda c: c.enable_stream(rs.stream.color, 424, 240, rs.format.bgr8, 15)))

    started = False
    for label, setup in attempts:
        cfg = rs.config()
        try:
            setup(cfg)
            pipeline.start(cfg)
            print(
                "[CAMERA] Intel RealSense %s → MJPEG /camera/rgb + /camera/depth（深度+彩色时）"
                % (label,)
            )
            started = True
            break
        except Exception as e:
            print("[CAMERA] RealSense 尝试 [%s] 失败: %s" % (label, e))
            _stop_safe()

    if not started:
        print(
            "[CAMERA] RealSense 全部配置失败。若已装 pyrealsense2 仍如此，多为设备未连接或固件/USB；"
            "可试 export FYP_REALSENSE_COLOR_ONLY=1 或换 USB3 口。"
        )
        return False
    while _cam_running:
        try:
            frames = pipeline.wait_for_frames(timeout_ms=5000)
        except Exception:
            time.sleep(0.05)
            continue
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue
        frame = np.asanyarray(color_frame.get_data())
        if frame.shape[1] != w or frame.shape[0] != h:
            frame = cv2.resize(frame, (w, h))
        ok_j, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
        if not ok_j:
            continue
        depth_frame = frames.get_depth_frame()
        with _cam_lock:
            _latest_jpeg = buf.tobytes()
            if depth_frame:
                dimg = np.asanyarray(depth_frame.get_data())
                if dimg.shape[1] != w or dimg.shape[0] != h:
                    dimg = cv2.resize(dimg, (w, h))
                depth_vis = _depth_uint16_to_colormap_bgr(dimg)
                ok_d, buf_d = cv2.imencode(
                    ".jpg",
                    depth_vis,
                    [int(cv2.IMWRITE_JPEG_QUALITY), 82],
                )
                _latest_depth_jpeg = buf_d.tobytes() if ok_d else None
            else:
                _latest_depth_jpeg = None
    try:
        pipeline.stop()
    except Exception:
        pass
    return True


def _camera_capture_loop():
    mode = FYP_CAMERA_MODE
    if mode == "uvc":
        _uvc_camera_loop()
        return
    if mode == "realsense":
        if not _realsense_camera_loop():
            print("[CAMERA] RealSense 不可用，回退 UVC…")
            _uvc_camera_loop()
        return
    # auto：与 jianjie 一致优先 RealSense（你的 D435）
    if _realsense_camera_loop():
        return
    print("[CAMERA] RealSense 未就绪，回退 OpenCV USB 摄像头（V4L2）…")
    _uvc_camera_loop()


def start_pi_camera_thread():
    global _cam_running, _cam_thread
    if not FYP_CAMERA_ENABLE:
        print("[CAMERA] 已通过 FYP_CAMERA_ENABLE=0 关闭摄像头采集")
        return
    if _cam_thread is not None:
        return
    _cam_running = True
    _cam_thread = threading.Thread(target=_camera_capture_loop, daemon=True)
    _cam_thread.start()
    print(
        "[CAMERA] 采集线程已启动（mode=%s）→ 上位机 RADAR_PI_BASE + USE_PI_CAMERA（config 已默认）"
        % (FYP_CAMERA_MODE or "auto",)
    )


def camera_mjpeg_generator():
    """multipart MJPEG，与上位机 OpenCV / 浏览器兼容。"""
    while True:
        with _cam_lock:
            chunk = _latest_jpeg
        if chunk:
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + chunk
                + b"\r\n"
            )
        time.sleep(1.0 / 24.0)


def camera_depth_mjpeg_generator():
    """深度伪彩 MJPEG；无深度流时用占位帧。"""
    while True:
        with _cam_lock:
            chunk = _latest_depth_jpeg
        out = chunk or _jpeg_depth_na()
        if out:
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + out
                + b"\r\n"
            )
        time.sleep(1.0 / 24.0)


@app.route("/camera/rgb")
def camera_rgb_stream():
    return Response(
        camera_mjpeg_generator(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/camera/depth")
def camera_depth_stream():
    return Response(
        camera_depth_mjpeg_generator(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


def _post_json(url, payload=b"{}"):
    headers = {"Content-Type": "application/json", "User-Agent": "shumeipai-pi/1.0"}
    if FYP_LAUNCHER_TOKEN:
        headers["X-FYP-Token"] = FYP_LAUNCHER_TOKEN
    req = urllib.request.Request(url, data=payload, method="POST", headers=headers)
    urllib.request.urlopen(req, timeout=120)


def _wait_until_ui_ready(ui_base, timeout_sec=120):
    if not ui_base:
        return False
    check = ui_base + "/api/status"
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        try:
            urllib.request.urlopen(check, timeout=2.0)
            return True
        except Exception:
            time.sleep(1.5)
    return False


def _open_chromium_kiosk(url):
    if not url:
        return
    cmds = [
        ["chromium-browser", "--kiosk", "--noerrdialogs", "--disable-infobars", url],
        ["chromium", "--kiosk", "--noerrdialogs", "--disable-infobars", url],
        ["google-chrome", "--kiosk", "--noerrdialogs", url],
    ]
    for cmd in cmds:
        try:
            subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
            print("[SUCCESS] 已尝试启动浏览器: %s" % cmd[0])
            return
        except Exception:
            continue
    print("[WARN] 无法启动 Chromium，请手动在显示器打开: %s" % url)


def startup_sequence():
    """唤醒上位机 face_app，等待就绪后在本机显示器打开统一前端。"""
    time.sleep(1.5)
    if FYP_LAUNCHER_URL:
        launch_url = FYP_LAUNCHER_URL + "/launch"
        try:
            print("[INFO] 请求上位机启动器: %s" % launch_url)
            _post_json(launch_url)
        except Exception as e:
            print("[WARN] 启动器调用失败（可改在本机先运行 fyp_launcher 与 face_app）: %s" % e)
    else:
        print("[INFO] 未设置 FYP_LAUNCHER_URL，跳过远程启动 face_app")

    if os.environ.get("FYP_OPEN_KIOSK", "1") != "1":
        return
    if not FYP_UI_URL:
        print("[INFO] 未设置 FYP_UI_URL，跳过自动打开浏览器")
        return
    if _wait_until_ui_ready(FYP_UI_URL):
        print("[SUCCESS] 上位机界面就绪: %s" % FYP_UI_URL)
        _open_chromium_kiosk(FYP_UI_URL)
    else:
        print("[WARN] 等待上位机超时，请手动打开: %s" % FYP_UI_URL)


class BreathingHeartSystem:
    """与 jianjie.py 相同的雷达与生命体征处理逻辑（树莓派无本地摄像头/Realsense）。"""

    def __init__(self, serial_port=None, baudrate=921600):
        self.serial_port = serial_port or SERIAL_PORT
        self.baudrate = baudrate
        self.serial_conn = None

        self.header = [0x66, 0x5B]
        self.data_length = 256
        self.sample_rate = 50
        self.radar_frequency = 24.125e9
        self.wavelength = 3e8 / self.radar_frequency

        self.breathing_freq_range = (0.1, 0.5)
        self.heartrate_freq_range = (1.0, 2.5)
        self.window_size = int(self.sample_rate * 30)

        # 主通道：与原逻辑一致（胸区平均，约 bin 100–156）
        self.phase_history = deque(maxlen=self.window_size)
        # 多距离门：将 0–256 分为多段，每段独立相位历史 → 多目标/多人距离分离（启发式）
        self.radar_channel_ranges = [
            (0, 51),
            (51, 102),
            (102, 153),
            (153, 204),
            (204, 256),
        ]
        self.phase_histories = [
            deque(maxlen=self.window_size) for _ in self.radar_channel_ranges
        ]
        self.radar_channels_snapshot = []
        self._ch_update_counter = 0
        self._ema_br = None
        self._ema_hr = None
        self._ema_ch = {}
        self._ema_alpha = 0.22
        self._br_raw_window = deque(maxlen=7)
        self._hr_raw_window = deque(maxlen=7)
        self.time_history = deque(maxlen=self.window_size)

        self.breathing_rate = 0.0
        self.breathing_amplitude = 0.0
        self.breathing_signal = []
        self.breathing_quality = 0.0

        self.heart_rate = 0.0
        self.heart_amplitude = 0.0
        self.heart_signal = []
        self.heart_quality = 0.0

        self.latest_phase_data = None
        self.frame_count = 0
        self.is_running = False
        self.start_time = None

        self.data_lock = threading.Lock()

        self.error_count = 0
        self.success_count = 0

        print("=" * 60)
        print("🫁 呼吸心跳检测系统（树莓派端 / 雷达本地）")
        print("雷达频率: %.3f GHz" % (self.radar_frequency / 1e9,))
        print("波长: %.2f mm" % (self.wavelength * 1000,))
        print(
            "呼吸检测: %s-%s Hz"
            % (self.breathing_freq_range[0], self.breathing_freq_range[1])
        )
        print(
            "心率检测: %s-%s Hz"
            % (self.heartrate_freq_range[0], self.heartrate_freq_range[1])
        )
        print("界面上位机（浏览器打开）: %s" % (FYP_UI_URL or "(请设置 FYP_UI_URL)"))
        print("=" * 60)

    def safe_filter_design(self, low_freq, high_freq, filter_order=4):
        try:
            nyquist = self.sample_rate / 2
            low_norm = max(0.001, min(0.999, low_freq / nyquist))
            high_norm = max(0.001, min(0.999, high_freq / nyquist))
            if low_norm >= high_norm:
                low_norm = high_norm * 0.8
            b, a = signal.butter(filter_order, [low_norm, high_norm], btype="band")
            if np.any(np.isnan(a)) or np.any(np.isnan(b)) or np.any(np.isinf(a)) or np.any(
                np.isinf(b)
            ):
                self.error_print("滤波器系数无效: low=%s, high=%s" % (low_norm, high_norm))
                return None, None
            return b, a
        except Exception as e:
            self.error_print("滤波器设计失败", e)
            return None, None

    def safe_filtfilt(self, b, a, data):
        try:
            if b is None or a is None:
                return data
            if np.any(np.isnan(data)) or np.any(np.isinf(data)):
                self.error_print("输入数据包含NaN或Inf")
                return np.zeros_like(data)
            filtered = signal.filtfilt(b, a, data)
            if np.any(np.isnan(filtered)) or np.any(np.isinf(filtered)):
                self.error_print("滤波输出包含NaN或Inf")
                return np.zeros_like(data)
            max_val = np.max(np.abs(filtered))
            if max_val > 1e10:
                self.error_print("滤波输出数值过大: %s" % (max_val,))
                filtered = filtered / (max_val / 1.0)
            return filtered
        except Exception as e:
            self.error_print("滤波过程错误", e)
            return np.zeros_like(data)

    def _compute_vitals_from_phase_array(self, phase_array, verbose_print=False):
        """由一段相位时间序列计算呼吸/心率及波形（不写主通道成员变量）。"""
        if len(phase_array) < self.sample_rate * 15:
            return None
        try:
            phase_array = np.asarray(phase_array, dtype=np.float64)
            if np.any(np.isnan(phase_array)) or np.any(np.isinf(phase_array)):
                return None
            phase_array = phase_array - np.mean(phase_array)
            phase_std = np.std(phase_array)
            if phase_std > 0:
                phase_array = np.clip(phase_array, -5 * phase_std, 5 * phase_std)

            b_breath, a_breath = self.safe_filter_design(0.1, 0.5, filter_order=4)
            if b_breath is not None:
                breathing_filtered = self.safe_filtfilt(b_breath, a_breath, phase_array)
            else:
                breathing_filtered = np.zeros_like(phase_array)

            b_heart, a_heart = self.safe_filter_design(1.0, 2.5, filter_order=4)
            if b_heart is not None:
                heart_filtered = self.safe_filtfilt(b_heart, a_heart, phase_array)
            else:
                heart_filtered = np.zeros_like(phase_array)

            if b_heart is not None:
                for harmonic_freq in [0.2, 0.4, 0.6, 0.8]:
                    try:
                        if harmonic_freq < 2.5:
                            nyquist = self.sample_rate / 2
                            notch_norm = harmonic_freq / nyquist
                            if 0.01 < notch_norm < 0.99:
                                Q = 10
                                b_notch, a_notch = signal.iirnotch(notch_norm, Q=Q)
                                heart_filtered = self.safe_filtfilt(
                                    b_notch, a_notch, heart_filtered
                                )
                    except Exception:
                        continue

            breathing_rate, breathing_amplitude = self.detect_frequency_in_range(
                breathing_filtered, 0.1, 0.5
            )
            heart_rate, heart_amplitude = self.detect_frequency_in_range(
                heart_filtered, 1.0, 2.5
            )

            if heart_rate < 40 or heart_rate > 180:
                heart_rate = 0.0
                heart_amplitude = 0.0

            if breathing_rate > 0 and abs(heart_rate - breathing_rate * 4) < 5:
                heart_rate = 0.0
                heart_amplitude = 0.0

            bq = self.calculate_signal_quality(breathing_filtered, breathing_amplitude)
            hq = self.calculate_signal_quality(heart_filtered, heart_amplitude)

            breathing_display = np.clip(breathing_filtered[-200:], -10, 10)
            heart_display = np.clip(heart_filtered[-200:], -10, 10)

            if verbose_print:
                print(
                    "检测结果 - 呼吸: %.1f次/分, 心率: %.1f次/分"
                    % (breathing_rate, heart_rate)
                )

            return {
                "breathing_rate": float(breathing_rate),
                "heart_rate": float(heart_rate),
                "breathing_amplitude": float(breathing_amplitude),
                "heart_amplitude": float(heart_amplitude),
                "breathing_quality": float(bq),
                "heart_quality": float(hq),
                "breathing_signal": breathing_display.tolist(),
                "heart_signal": heart_display.tolist(),
            }
        except Exception as e:
            self.error_print("生命体征检测错误", e)
            return None

    def _limit_rate_jump(self, prev_display: float, target: float, max_step: float):
        """单帧输出限幅，抑制 FFT 尖峰导致的 BPM 猛跳。"""
        try:
            pv = float(prev_display)
            tv = float(target)
        except (TypeError, ValueError):
            return float(target)
        if pv <= 0 or math.isnan(pv):
            return tv
        if math.isnan(tv):
            return pv
        d = tv - pv
        if abs(d) <= max_step:
            return tv
        return pv + math.copysign(max_step, d)

    def detect_vital_signs_with_stable_filtering(self):
        if len(self.phase_history) < self.sample_rate * 15:
            return None
        phase_array = np.array(list(self.phase_history), dtype=np.float64)
        res = self._compute_vitals_from_phase_array(phase_array, verbose_print=True)
        if res is None:
            return None
        self._br_raw_window.append(float(res["breathing_rate"]))
        self._hr_raw_window.append(float(res["heart_rate"]))
        br_in = float(np.median(self._br_raw_window))
        hr_in = float(np.median(self._hr_raw_window))
        self._ema_br = self._ema_smooth_scalar(self._ema_br, br_in)
        self._ema_hr = self._ema_smooth_scalar(self._ema_hr, hr_in)
        self.breathing_rate = self._limit_rate_jump(self.breathing_rate, self._ema_br, 1.8)
        self.heart_rate = self._limit_rate_jump(self.heart_rate, self._ema_hr, 4.0)
        self.breathing_amplitude = res["breathing_amplitude"]
        self.heart_amplitude = res["heart_amplitude"]
        self.breathing_quality = res["breathing_quality"]
        self.heart_quality = res["heart_quality"]
        self.breathing_signal = res["breathing_signal"]
        self.heart_signal = res["heart_signal"]
        return {
            "breathing_signal": self.breathing_signal,
            "heart_signal": self.heart_signal,
            "breathing_rate_bpm": float(self.breathing_rate),
            "heart_rate_bpm": float(self.heart_rate),
            "breathing_quality": float(self.breathing_quality),
            "heart_quality": float(self.heart_quality),
        }

    def _refresh_radar_channels_snapshot(self):
        """更新多距离门快照（供 /api/radar 返回）。"""
        snap = []
        for idx, (a, b) in enumerate(self.radar_channel_ranges):
            hist = self.phase_histories[idx]
            if len(hist) < self.sample_rate * 15:
                snap.append(
                    {
                        "id": idx,
                        "bin_start": a,
                        "bin_end": b,
                        "breathing_rate": 0.0,
                        "heart_rate": 0.0,
                        "breathing_quality": 0.0,
                        "heart_quality": 0.0,
                        "ready": False,
                        "breathing_signal": [],
                        "heart_signal": [],
                    }
                )
                continue
            arr = np.array(list(hist), dtype=np.float64)
            res = self._compute_vitals_from_phase_array(arr, verbose_print=False)
            if res is None:
                snap.append(
                    {
                        "id": idx,
                        "bin_start": a,
                        "bin_end": b,
                        "breathing_rate": 0.0,
                        "heart_rate": 0.0,
                        "breathing_quality": 0.0,
                        "heart_quality": 0.0,
                        "ready": False,
                        "breathing_signal": [],
                        "heart_signal": [],
                    }
                )
                continue
            br_raw = float(res["breathing_rate"])
            hr_raw = float(res["heart_rate"])
            prev = self._ema_ch.get(idx, (None, None))
            br_s = self._ema_smooth_scalar(prev[0], br_raw)
            hr_s = self._ema_smooth_scalar(prev[1], hr_raw)
            self._ema_ch[idx] = (br_s, hr_s)
            br_sig = res.get("breathing_signal") or []
            hr_sig = res.get("heart_signal") or []
            snap.append(
                {
                    "id": idx,
                    "bin_start": a,
                    "bin_end": b,
                    "breathing_rate": round(br_s, 1),
                    "heart_rate": round(hr_s, 1),
                    "breathing_quality": round(res["breathing_quality"], 3),
                    "heart_quality": round(res["heart_quality"], 3),
                    "ready": True,
                    "breathing_signal": br_sig[-100:] if len(br_sig) > 100 else list(br_sig),
                    "heart_signal": hr_sig[-100:] if len(hr_sig) > 100 else list(hr_sig),
                }
            )
        self.radar_channels_snapshot = snap

    def _ema_smooth_scalar(self, prev, new_val):
        """指数平滑，抑制瞬时频谱跳变；new<=0 时保持上一有效值。"""
        a = self._ema_alpha
        try:
            nv = float(new_val)
        except (TypeError, ValueError):
            nv = 0.0
        if nv <= 0 or math.isnan(nv):
            return prev if prev is not None else 0.0
        if prev is None or prev <= 0 or math.isnan(prev):
            return nv
        return a * nv + (1.0 - a) * float(prev)

    def _refined_fft_peak_bpm(self, signal_data, low_freq, high_freq):
        """Hann 窗 + rFFT；频带内峰值；对数域抛物线插值细化主频。"""
        if len(signal_data) < 64:
            return 0.0, 0.0
        x = np.asarray(signal_data, dtype=np.float64)
        if np.all(x == 0) or np.std(x) < 1e-10:
            return 0.0, 0.0
        n = len(x)
        fs = float(self.sample_rate)
        win = np.hanning(n)
        xw = x * win
        mag = np.abs(np.fft.rfft(xw))
        freqs = np.fft.rfftfreq(n, 1.0 / fs)
        mask = (freqs >= low_freq) & (freqs <= high_freq)
        if not np.any(mask):
            return 0.0, 0.0
        sm = mag[mask]
        if len(sm) == 0 or np.max(sm) <= 0:
            return 0.0, 0.0
        peak_rel = int(np.argmax(sm))
        gidx = int(np.flatnonzero(mask)[peak_rel])
        amplitude = float(sm[peak_rel])
        if gidx <= 0 or gidx >= len(mag) - 1:
            f_hz = float(freqs[gidx])
        else:
            y0 = math.log(mag[gidx - 1] + 1e-12)
            y1 = math.log(mag[gidx] + 1e-12)
            y2 = math.log(mag[gidx + 1] + 1e-12)
            denom = y0 - 2.0 * y1 + y2
            if abs(denom) < 1e-9:
                f_hz = float(gidx) * fs / float(n)
            else:
                delta = 0.5 * (y0 - y2) / denom
                delta = float(np.clip(delta, -0.5, 0.5))
                f_hz = (float(gidx) + delta) * fs / float(n)
        return float(f_hz * 60.0), amplitude

    def _autocorr_rate_bpm(self, signal_data, low_freq, high_freq):
        """归一化自相关在生理滞后窗内找主峰，与 FFT 互补（抑制泄漏 / 谐波误判）。"""
        x = np.asarray(signal_data, dtype=np.float64)
        x = x - np.mean(x)
        n = len(x)
        if n < 96:
            return None
        fs = float(self.sample_rate)
        lag_min = max(1, int(fs / high_freq))
        lag_max = min(n // 2 - 1, int(fs / low_freq))
        if lag_max <= lag_min + 2:
            return None
        nfft = int(2 ** math.ceil(math.log2(2 * n - 1)))
        X = np.fft.rfft(x, n=nfft)
        ac = np.fft.irfft(X * np.conj(X))[:n]
        if ac[0] <= 1e-12:
            return None
        ac = ac / ac[0]
        seg = ac[lag_min : lag_max + 1]
        if len(seg) < 3:
            return None
        lag = lag_min + int(np.argmax(seg))
        if lag <= 0:
            return None
        return float(60.0 * fs / float(lag))

    def _blend_fft_and_autocorr_bpm(self, fft_bpm, fft_amp, ac_bpm, low_f, high_f):
        """FFT 与自相关折中：一致则偏向自相关平滑；分歧大则以 FFT 为主并保留少量 AC。"""
        if fft_bpm <= 0 and ac_bpm is not None and ac_bpm > 0:
            return float(ac_bpm), float(fft_amp)
        if fft_bpm <= 0:
            return 0.0, 0.0
        if ac_bpm is None or ac_bpm <= 0:
            return float(fft_bpm), float(fft_amp)
        m = max(fft_bpm, ac_bpm, 1.0)
        rel = abs(fft_bpm - ac_bpm) / m
        if rel <= 0.11:
            w_fft, w_ac = 0.52, 0.48
        elif rel <= 0.22:
            w_fft, w_ac = 0.62, 0.38
        else:
            w_fft, w_ac = 0.72, 0.28
        lo_bpm = low_f * 60.0
        hi_bpm = high_f * 60.0
        blend = w_fft * fft_bpm + w_ac * ac_bpm
        blend = float(np.clip(blend, lo_bpm * 0.85, hi_bpm * 1.15))
        return blend, float(fft_amp)

    def detect_frequency_in_range(self, signal_data, low_freq, high_freq):
        fft_bpm, fft_amp = self._refined_fft_peak_bpm(signal_data, low_freq, high_freq)
        ac_bpm = self._autocorr_rate_bpm(signal_data, low_freq, high_freq)
        return self._blend_fft_and_autocorr_bpm(
            fft_bpm, fft_amp, ac_bpm, low_freq, high_freq
        )

    def calculate_signal_quality(self, signal_data, amplitude):
        if len(signal_data) == 0 or amplitude == 0:
            return 0.0
        try:
            if np.all(signal_data == 0):
                return 0.0
            signal_std = np.std(signal_data)
            if signal_std == 0:
                return 0.0
            noise_level = np.std(np.diff(signal_data))
            snr = signal_std / (noise_level + 1e-6)
            quality = min(1.0, max(0.0, snr / 10.0))
            return quality
        except Exception:
            return 0.0

    def process_radar_data(self, raw_data):
        try:
            byte_array = np.frombuffer(raw_data, dtype=np.uint8)
            phase_data = byte_array * (2 * np.pi / 255.0)
            phase_unwrapped = np.unwrap(phase_data)
            chest_region = phase_unwrapped[100:156]
            phase_mean = np.mean(chest_region)
            if np.isnan(phase_mean) or np.isinf(phase_mean):
                self.error_print("相位数据包含异常值")
                return None
            current_time = time.time()
            self.phase_history.append(phase_mean)
            self.time_history.append(current_time)
            for idx, (a, b) in enumerate(self.radar_channel_ranges):
                lo, hi = max(0, a), min(256, b)
                if hi <= lo:
                    continue
                seg = phase_unwrapped[lo:hi]
                ch_mean = float(np.mean(seg))
                if np.isnan(ch_mean) or np.isinf(ch_mean):
                    continue
                self.phase_histories[idx].append(ch_mean)
            self._ch_update_counter += 1
            if self._ch_update_counter % 4 == 0:
                self._refresh_radar_channels_snapshot()
            vital_signs = self.detect_vital_signs_with_stable_filtering()
            return {
                "phase_unwrapped": phase_unwrapped.tolist(),
                "phase_mean": float(phase_mean),
                "breathing_rate": float(self.breathing_rate),
                "heart_rate": float(self.heart_rate),
                "breathing_quality": float(self.breathing_quality),
                "heart_quality": float(self.heart_quality),
                "vital_signs": vital_signs,
            }
        except Exception as e:
            self.error_print("数据处理错误", e)
            return None

    def connect_serial(self):
        try:
            self.serial_conn = serial.Serial(
                port=self.serial_port, baudrate=self.baudrate, timeout=0.1
            )
            self.success_print("串口连接成功: %s" % (self.serial_port,))
            return True
        except Exception as e:
            self.error_print("串口连接失败", e)
            return False

    def read_radar_frame(self):
        if not self.serial_conn:
            return None
        try:
            data = self.serial_conn.read(self.data_length + 2)
            if len(data) >= self.data_length:
                return data[-self.data_length :]
            return None
        except Exception:
            return None

    def radar_reader_thread(self):
        self.success_print("启动雷达读取线程")
        while self.is_running:
            try:
                raw_data = self.read_radar_frame()
                if raw_data is None:
                    time.sleep(0.01)
                    continue
                processed_data = self.process_radar_data(raw_data)
                if processed_data:
                    self.frame_count += 1
                    with self.data_lock:
                        self.latest_phase_data = processed_data
            except Exception as e:
                self.error_print("雷达读取错误", e)
                time.sleep(0.1)

    def start_radar(self):
        if not self.connect_serial():
            return False
        self.is_running = True
        self.start_time = time.time()
        self.radar_thread = threading.Thread(target=self.radar_reader_thread)
        self.radar_thread.daemon = True
        self.radar_thread.start()
        self.success_print("雷达系统已启动")
        return True

    def stop_radar(self):
        self.is_running = False
        self.breathing_rate = 0.0
        self.heart_rate = 0.0
        self.breathing_amplitude = 0.0
        self.heart_amplitude = 0.0
        self.breathing_quality = 0.0
        self.heart_quality = 0.0
        self.breathing_signal = []
        self.heart_signal = []
        self.phase_history.clear()
        for d in self.phase_histories:
            d.clear()
        self.radar_channels_snapshot = []
        self._ch_update_counter = 0
        self._ema_br = None
        self._ema_hr = None
        self._ema_ch.clear()
        self._br_raw_window.clear()
        self._hr_raw_window.clear()
        self.time_history.clear()
        with self.data_lock:
            self.latest_phase_data = None
        self.frame_count = 0
        if self.serial_conn:
            self.serial_conn.close()
            self.serial_conn = None
        self.success_print("雷达已停止")

    def get_latest_data(self):
        current_time = time.time()
        fps = 0
        if self.start_time and self.is_running:
            elapsed = current_time - self.start_time
            fps = self.frame_count / elapsed if elapsed > 0 else 0
        with self.data_lock:
            current_phase = self.latest_phase_data
        if not self.is_running:
            return {
                "frame_count": 0,
                "fps": 0.0,
                "current_phase": None,
                "is_running": False,
                "breathing_rate": 0.0,
                "heart_rate": 0.0,
                "breathing_quality": 0.0,
                "heart_quality": 0.0,
                "breathing_signal": [],
                "heart_signal": [],
                "radar_channels": [],
            }
        return {
            "frame_count": int(self.frame_count),
            "fps": float(round(fps, 1)),
            "current_phase": current_phase,
            "is_running": bool(self.is_running),
            "breathing_rate": float(round(self.breathing_rate, 1)),
            "heart_rate": float(round(self.heart_rate, 1)),
            "breathing_quality": float(round(self.breathing_quality, 3)),
            "heart_quality": float(round(self.heart_quality, 3)),
            "breathing_signal": self.breathing_signal[-100:] if self.breathing_signal else [],
            "heart_signal": self.heart_signal[-100:] if self.heart_signal else [],
            "radar_channels": list(self.radar_channels_snapshot),
        }

    def error_print(self, message, exception=None):
        timestamp = datetime.now().strftime("%H:%M:%S")
        print("[%s] ERROR: %s" % (timestamp, message))
        if exception:
            print("Exception: %s" % (str(exception),))
        self.error_count += 1

    def success_print(self, message):
        timestamp = datetime.now().strftime("%H:%M:%S")
        print("[%s] SUCCESS: %s" % (timestamp, message))
        self.success_count += 1


breathing_system = BreathingHeartSystem()


@app.route("/")
def index():
    ui = FYP_UI_URL or "(请设置环境变量 FYP_UI_URL 为上位机 face_app 地址)"
    cam_note = (
        "摄像头 MJPEG：<code>/camera/rgb</code>、<code>/camera/depth</code>（RealSense 深度+彩色时）"
        if FYP_CAMERA_ENABLE
        else "摄像头已关闭（FYP_CAMERA_ENABLE=0）"
    )
    return (
        """<!DOCTYPE html><meta charset="utf-8"><title>树莓派采集端</title>
<body style="font-family:sans-serif;padding:24px;max-width:720px;line-height:1.6;">
<h2>树莓派采集端（雷达串口 + USB 摄像头）</h2>
<p>请在显示器使用 Chromium 打开<strong>上位机</strong>统一界面：</p>
<p><strong>%s</strong></p>
<p>%s</p>
<p>雷达接口：<code>/api/radar</code>、<code>/api/start-radar</code>、<code>/api/stop-radar</code></p>
</body>"""
        % (ui, cam_note)
    )


@app.route("/api/radar")
def api_radar():
    """供上位机 face_app 轮询，勿在此再请求上位机（避免循环）。"""
    return jsonify(breathing_system.get_latest_data())


@app.route("/api/start-radar", methods=["POST"])
def start_radar():
    try:
        success = breathing_system.start_radar()
        return jsonify({"success": success, "message": "雷达已启动" if success else "启动失败"})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)})


@app.route("/api/stop-radar", methods=["POST"])
def stop_radar():
    try:
        breathing_system.stop_radar()
        return jsonify({"success": True, "message": "雷达已停止"})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)})


if __name__ == "__main__":
    print("树莓派采集服务监听 %s:%s" % (HTTP_HOST, HTTP_PORT))
    print("上位机示例：RADAR_PI_BASE=http://10.162.133.43:%s（换网段请改 IP）" % HTTP_PORT)
    print("若摄像头插在树莓派：上位机设 USE_PI_CAMERA=1（并确保本机已开启摄像头采集）")
    start_pi_camera_thread()
    if os.environ.get("FYP_AUTO_START", "1") == "1":
        threading.Thread(target=startup_sequence, daemon=True).start()
    try:
        app.run(host=HTTP_HOST, port=HTTP_PORT, debug=False, threaded=True)
    except KeyboardInterrupt:
        print("\n正在关闭...")
        breathing_system.stop_radar()
        print("已退出")
