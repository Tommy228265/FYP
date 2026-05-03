#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
树莓派端：仅运行毫米波雷达算法（与 jianjie.py 一致）并提供 HTTP API。
完整前端界面在上位机 templates/index.html（face_app）；显示器浏览器打开上位机地址。
启动时可 POST 唤醒上位机 fyp_launcher.py 并尝试全屏打开 Chromium。

环境变量（均有默认，重启后也可直接: python3 shumeipai.py）：
  FYP_PC_HOST=10.162.133.140   # 可选；不设则用下方代码内 _DEFAULT_PC_HOST 自动生成 UI/Launcher URL
  FYP_UI_URL / FYP_LAUNCHER_URL  # 可选；单独指定则覆盖由 PC_HOST 拼出的地址
  FYP_LAUNCHER_TOKEN=your_secret
  FYP_OPEN_KIOSK=1   FYP_AUTO_START=1   FYP_SERIAL=/dev/ttyUSB0
"""

import math
import os
import serial
import subprocess
import threading
import time
import urllib.request
from collections import deque
from datetime import datetime

import numpy as np
from flask import Flask, jsonify
from scipy import signal

# 上位机（笔记本）局域网 IPv4；换网络或电脑 IP 变了时只改这一处即可。
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

app = Flask(__name__)


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
        self._ema_alpha = 0.26
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

    def detect_vital_signs_with_stable_filtering(self):
        if len(self.phase_history) < self.sample_rate * 15:
            return None
        phase_array = np.array(list(self.phase_history), dtype=np.float64)
        res = self._compute_vitals_from_phase_array(phase_array, verbose_print=True)
        if res is None:
            return None
        self._ema_br = self._ema_smooth_scalar(self._ema_br, res["breathing_rate"])
        self._ema_hr = self._ema_smooth_scalar(self._ema_hr, res["heart_rate"])
        self.breathing_rate = self._ema_br
        self.heart_rate = self._ema_hr
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
                    }
                )
                continue
            br_raw = float(res["breathing_rate"])
            hr_raw = float(res["heart_rate"])
            prev = self._ema_ch.get(idx, (None, None))
            br_s = self._ema_smooth_scalar(prev[0], br_raw)
            hr_s = self._ema_smooth_scalar(prev[1], hr_raw)
            self._ema_ch[idx] = (br_s, hr_s)
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
            w_fft, w_ac = 0.68, 0.32
        else:
            w_fft, w_ac = 0.82, 0.18
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
    return (
        """<!DOCTYPE html><meta charset="utf-8"><title>树莓派雷达 API</title>
<body style="font-family:sans-serif;padding:24px;">
<h2>树莓派仅提供雷达串口与 API</h2>
<p>请在显示器使用 Chromium 打开上位机统一界面：</p>
<p><strong>%s</strong></p>
<p>接口：<code>/api/radar</code>、<code>/api/start-radar</code>、<code>/api/stop-radar</code></p>
</body>"""
        % (ui,)
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
    print("树莓派雷达服务监听 %s:%s" % (HTTP_HOST, HTTP_PORT))
    print("为上位机配置 RADAR_PI_BASE=http://<树莓派IP>:%s" % HTTP_PORT)
    if os.environ.get("FYP_AUTO_START", "1") == "1":
        threading.Thread(target=startup_sequence, daemon=True).start()
    try:
        app.run(host=HTTP_HOST, port=HTTP_PORT, debug=False, threaded=True)
    except KeyboardInterrupt:
        print("\n正在关闭...")
        breathing_system.stop_radar()
        print("已退出")
