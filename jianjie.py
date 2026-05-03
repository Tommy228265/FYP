#!/usr/bin/env python3
"""
呼吸心跳检测系统
基于24GHz毫米波雷达的非接触式生命体征监测
"""

import serial
import threading
import time
import numpy as np
import cv2
import pyrealsense2 as rs
from datetime import datetime
from flask import Flask, Response, jsonify
from collections import deque
from scipy import signal
from PIL import Image, ImageDraw, ImageFont
import sys
import os

app = Flask(__name__)

class BreathingHeartSystem:
    def __init__(self, serial_port='/dev/ttyUSB0', baudrate=921600):
        self.serial_port = serial_port
        self.baudrate = baudrate
        self.serial_conn = None
        
        # 雷达参数
        self.header = [0x66, 0x5B]
        self.data_length = 256
        self.sample_rate = 50
        self.radar_frequency = 24.125e9
        self.wavelength = 3e8 / self.radar_frequency
        
        # 生命体征参数
        self.breathing_freq_range = (0.1, 0.5)
        self.heartrate_freq_range = (1.0, 2.5)
        self.window_size = int(self.sample_rate * 30)
        
        # 数据存储
        self.phase_history = deque(maxlen=self.window_size)
        self.time_history = deque(maxlen=self.window_size)
        
        # 生命体征结果
        self.breathing_rate = 0.0
        self.breathing_amplitude = 0.0
        self.breathing_signal = []
        self.breathing_quality = 0.0
        
        self.heart_rate = 0.0
        self.heart_amplitude = 0.0
        self.heart_signal = []
        self.heart_quality = 0.0
        
        # 人脸检测参数
        self.face_cascade = None
        self.font = None
        self.THRESHOLD_DISTANCE = 1.0
        self.FONT_PATH = "/usr/share/fonts/truetype/freefont/FreeSans.ttf"
        
        # 摄像头参数
        self.camera_pipeline = None
        self.camera_running = False
        self.align = None
        self.current_distance = 0.0
        self.person_detected = False
        self.face_detected = False
        self.distance_warning = False
        self.face_count = 0
        self.face_positions = []
        
        # 图像缓存
        self.latest_color_frame = None
        self.latest_depth_frame = None
        self.latest_face_frame = None
        
        # 系统状态
        self.latest_phase_data = None
        self.frame_count = 0
        self.is_running = False
        self.start_time = None
        
        # 线程锁
        self.frame_lock = threading.Lock()
        self.data_lock = threading.Lock()
        self.face_lock = threading.Lock()
        
        # 统计信息
        self.error_count = 0
        self.success_count = 0
        
        print("=" * 60)
        print(f"🫁 呼吸心跳检测系统")
        print(f"雷达频率: {self.radar_frequency/1e9:.3f} GHz")
        print(f"波长: {self.wavelength*1000:.2f} mm")
        print(f"呼吸检测: {self.breathing_freq_range[0]}-{self.breathing_freq_range[1]} Hz")
        print(f"心率检测: {self.heartrate_freq_range[0]}-{self.heartrate_freq_range[1]} Hz")
        print("=" * 60)
    
    def safe_filter_design(self, low_freq, high_freq, filter_order=4):
        """安全的滤波器设计"""
        try:
            nyquist = self.sample_rate / 2
            
            # 确保频率在有效范围内
            low_norm = max(0.001, min(0.999, low_freq / nyquist))
            high_norm = max(0.001, min(0.999, high_freq / nyquist))
            
            # 确保 low < high
            if low_norm >= high_norm:
                low_norm = high_norm * 0.8
            
            # 使用较低的阶数提高稳定性
            b, a = signal.butter(filter_order, [low_norm, high_norm], btype='band')
            
            # 检查滤波器系数是否有效
            if np.any(np.isnan(a)) or np.any(np.isnan(b)) or np.any(np.isinf(a)) or np.any(np.isinf(b)):
                self.error_print(f"滤波器系数无效: low={low_norm}, high={high_norm}")
                return None, None
            
            return b, a
            
        except Exception as e:
            self.error_print(f"滤波器设计失败: {e}")
            return None, None
    
    def safe_filtfilt(self, b, a, data):
        """安全的滤波函数"""
        try:
            if b is None or a is None:
                return data
            
            # 检查输入数据
            if np.any(np.isnan(data)) or np.any(np.isinf(data)):
                self.error_print("输入数据包含NaN或Inf")
                return np.zeros_like(data)
            
            # 应用滤波器
            filtered = signal.filtfilt(b, a, data)
            
            # 检查输出数据
            if np.any(np.isnan(filtered)) or np.any(np.isinf(filtered)):
                self.error_print("滤波输出包含NaN或Inf")
                return np.zeros_like(data)
            
            # 检查数值范围，防止溢出
            max_val = np.max(np.abs(filtered))
            if max_val > 1e10:  # 如果值太大
                self.error_print(f"滤波输出数值过大: {max_val}")
                # 归一化到合理范围
                filtered = filtered / (max_val / 1.0)
            
            return filtered
            
        except Exception as e:
            self.error_print(f"滤波过程错误: {e}")
            return np.zeros_like(data)
    
    def detect_vital_signs_with_stable_filtering(self):
        """生命体征检测"""
        if len(self.phase_history) < self.sample_rate * 15:
            return None
        
        try:
            # 使用float64提高精度
            phase_array = np.array(list(self.phase_history), dtype=np.float64)
            
            # 检查原始数据
            if np.any(np.isnan(phase_array)) or np.any(np.isinf(phase_array)):
                self.error_print("原始相位数据包含异常值")
                return None
            
            # 去除直流分量
            phase_array = phase_array - np.mean(phase_array)
            
            # 限制数据范围，防止极值
            phase_std = np.std(phase_array)
            if phase_std > 0:
                phase_array = np.clip(phase_array, -5 * phase_std, 5 * phase_std)
            
            # 设计稳定的呼吸滤波器
            b_breath, a_breath = self.safe_filter_design(0.1, 0.5, filter_order=4)
            if b_breath is not None:
                breathing_filtered = self.safe_filtfilt(b_breath, a_breath, phase_array)
            else:
                breathing_filtered = np.zeros_like(phase_array)
            
            # 设计稳定的心率滤波器
            b_heart, a_heart = self.safe_filter_design(1.0, 2.5, filter_order=4)
            if b_heart is not None:
                heart_filtered = self.safe_filtfilt(b_heart, a_heart, phase_array)
            else:
                heart_filtered = np.zeros_like(phase_array)
            
            # 额外的陷波滤波器（仅在心率滤波成功时应用）
            if b_heart is not None:
                for harmonic_freq in [0.2, 0.4, 0.6, 0.8]:
                    try:
                        if harmonic_freq < 2.5:
                            nyquist = self.sample_rate / 2
                            notch_norm = harmonic_freq / nyquist
                            if 0.01 < notch_norm < 0.99:  # 更保守的范围
                                Q = 10  # 降低Q值，提高稳定性
                                b_notch, a_notch = signal.iirnotch(notch_norm, Q=Q)
                                heart_filtered = self.safe_filtfilt(b_notch, a_notch, heart_filtered)
                    except Exception as e:
                        continue
            
            # 频率检测
            breathing_rate, breathing_amplitude = self.detect_frequency_in_range(
                breathing_filtered, 0.1, 0.5
            )
            heart_rate, heart_amplitude = self.detect_frequency_in_range(
                heart_filtered, 1.0, 2.5
            )
            
            # 心率合理性检查
            if heart_rate < 40 or heart_rate > 180:
                heart_rate = 0.0
                heart_amplitude = 0.0
            
            # 检查心率是否是呼吸的谐波
            if breathing_rate > 0 and abs(heart_rate - breathing_rate * 4) < 5:
                heart_rate = 0.0
                heart_amplitude = 0.0
            
            # 更新基本结果
            self.breathing_rate = breathing_rate
            self.heart_rate = heart_rate
            self.breathing_amplitude = breathing_amplitude
            self.heart_amplitude = heart_amplitude
            
            # 计算质量
            self.breathing_quality = self.calculate_signal_quality(breathing_filtered, breathing_amplitude)
            self.heart_quality = self.calculate_signal_quality(heart_filtered, heart_amplitude)
            
            # 存储波形（限制长度和数值范围）
            breathing_display = breathing_filtered[-200:]
            heart_display = heart_filtered[-200:]
            
            # 最终数值检查和范围限制
            breathing_display = np.clip(breathing_display, -10, 10)
            heart_display = np.clip(heart_display, -10, 10)
            
            self.breathing_signal = breathing_display.tolist()
            self.heart_signal = heart_display.tolist()
            
            print(f"检测结果 - 呼吸: {breathing_rate:.1f}次/分, 心率: {heart_rate:.1f}次/分")
            
            return {
                'breathing_signal': self.breathing_signal,
                'heart_signal': self.heart_signal,
                'breathing_rate_bpm': float(breathing_rate),
                'heart_rate_bpm': float(heart_rate),
                'breathing_quality': float(self.breathing_quality),
                'heart_quality': float(self.heart_quality)
            }
            
        except Exception as e:
            self.error_print("生命体征检测错误", e)
            return None
    
    def init_face_detection(self):
        """初始化人脸检测"""
        try:
            cascade_path = '/home/pi/models/haarcascade_frontalface_default.xml'
            if not os.path.exists(cascade_path):
                cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
            
            if self.face_cascade.empty():
                self.error_print("无法加载人脸检测模型")
                return False
            
            try:
                self.font = ImageFont.truetype(self.FONT_PATH, 20)
            except:
                self.error_print("无法加载字体，使用默认字体")
                self.font = ImageFont.load_default()
            
            self.success_print("人脸检测初始化成功")
            return True
            
        except Exception as e:
            self.error_print("人脸检测初始化失败", e)
            return False
    
    def init_camera_with_face_detection(self):
        """初始化摄像头"""
        try:
            if not self.init_face_detection():
                return False
            
            self.camera_pipeline = rs.pipeline()
            config = rs.config()
            config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
            config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            profile = self.camera_pipeline.start(config)
            
            self.align = rs.align(rs.stream.color)
            
            self.success_print("摄像头初始化成功")
            return True
            
        except Exception as e:
            self.error_print("摄像头初始化失败", e)
            return False
    
    def detect_frequency_in_range(self, signal_data, low_freq, high_freq):
        """检测指定频率范围内的主导频率"""
        if len(signal_data) < 64:
            return 0.0, 0.0
        
        try:
            # 检查信号是否全为零
            if np.all(signal_data == 0) or np.std(signal_data) < 1e-10:
                return 0.0, 0.0
            
            fft_data = np.fft.fft(signal_data)
            freqs = np.fft.fftfreq(len(signal_data), 1/self.sample_rate)
            magnitude = np.abs(fft_data)
            
            positive_freqs = freqs[:len(freqs)//2]
            positive_magnitude = magnitude[:len(magnitude)//2]
            
            freq_mask = (positive_freqs >= low_freq) & (positive_freqs <= high_freq)
            
            if np.any(freq_mask):
                masked_freqs = positive_freqs[freq_mask]
                masked_magnitude = positive_magnitude[freq_mask]
                
                if len(masked_magnitude) > 0 and np.max(masked_magnitude) > 0:
                    peak_idx = np.argmax(masked_magnitude)
                    dominant_freq = masked_freqs[peak_idx]
                    amplitude = masked_magnitude[peak_idx]
                    
                    rate_per_minute = dominant_freq * 60
                    return rate_per_minute, amplitude
            
            return 0.0, 0.0
            
        except Exception as e:
            self.error_print("频率检测错误", e)
            return 0.0, 0.0
    
    def calculate_signal_quality(self, signal_data, amplitude):
        """计算信号质量"""
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
        except:
            return 0.0
    
    def detect_faces_and_distance(self, color_frame, depth_frame):
        """检测人脸和距离"""
        try:
            color_image = np.asanyarray(color_frame.get_data())
            
            gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.2, 5)
            
            color_rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(color_rgb)
            draw = ImageDraw.Draw(pil_image)
            
            face_info = []
            total_distance = 0
            valid_faces = 0
            
            for (x, y, w, h) in faces:
                cx, cy = x + w // 2, y + h // 2
                distance = depth_frame.get_distance(cx, cy)
                
                if distance > 0:
                    valid_faces += 1
                    total_distance += distance
                    
                    if distance > self.THRESHOLD_DISTANCE:
                        label = "距离过远"
                        color = (255, 0, 0)
                        warning = True
                    else:
                        label = "距离合格"
                        color = (0, 255, 0)
                        warning = False
                    
                    face_info.append({
                        'x': x, 'y': y, 'w': w, 'h': h,
                        'distance': distance,
                        'warning': warning,
                        'center': (cx, cy)
                    })
                    
                    print(f"检测到人脸，距离：{distance:.2f} 米")
                    
                    cv2.rectangle(color_image, (x, y), (x+w, y+h), color, 2)
                    draw.text((x, y - 25 if y > 30 else y + h + 5), 
                             f"{label} {distance:.2f}m", font=self.font, fill=color)
            
            with self.face_lock:
                self.face_count = len(faces)
                self.face_detected = len(faces) > 0
                self.face_positions = face_info
                
                if valid_faces > 0:
                    self.current_distance = total_distance / valid_faces
                    self.person_detected = True
                    self.distance_warning = any(face['warning'] for face in face_info)
                else:
                    self.current_distance = 0.0
                    self.person_detected = False
                    self.distance_warning = False
            
            color_display = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            
            return color_display, len(faces)
            
        except Exception as e:
            self.error_print("人脸检测错误", e)
            return color_image, 0
    
    def camera_stream_thread(self):
        """摄像头流线程"""
        self.success_print("启动人脸检测摄像头流")
        
        frame_count = 0
        last_time = time.time()
        
        while self.camera_running:
            try:
                frames = self.camera_pipeline.wait_for_frames()
                aligned_frames = self.align.process(frames)
                depth_frame = aligned_frames.get_depth_frame()
                color_frame = aligned_frames.get_color_frame()
                
                if not depth_frame or not color_frame:
                    continue
                
                frame_count += 1
                
                color_image = np.asanyarray(color_frame.get_data())
                depth_image = np.asanyarray(depth_frame.get_data())
                
                face_image, face_count = self.detect_faces_and_distance(color_frame, depth_frame)
                
                depth_colormap = cv2.applyColorMap(
                    cv2.convertScaleAbs(depth_image, alpha=0.03), 
                    cv2.COLORMAP_JET
                )
                
                color_with_info = self.add_breathing_overlay(color_image.copy())
                depth_with_info = self.add_breathing_overlay(depth_colormap.copy())
                
                with self.frame_lock:
                    self.latest_color_frame = color_with_info
                    self.latest_depth_frame = depth_with_info
                    self.latest_face_frame = face_image
                
                current_time = time.time()
                if current_time - last_time >= 3.0:
                    fps = frame_count / (current_time - last_time)
                    self.success_print(f"摄像头FPS: {fps:.1f} | 人脸: {self.face_count}")
                    frame_count = 0
                    last_time = current_time
                
            except Exception as e:
                self.error_print("摄像头流错误", e)
                time.sleep(0.1)
    
    def add_breathing_overlay(self, image):
        """添加呼吸信息覆盖"""
        try:
            height, width = image.shape[:2]
            
            overlay = image.copy()
            cv2.rectangle(overlay, (10, 10), (width-10, 200), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)
            
            y_pos = 30
            cv2.putText(image, f"Breathing: {self.breathing_rate:.1f} bpm", 
                       (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            y_pos += 35
            cv2.putText(image, f"Heart Rate: {self.heart_rate:.1f} bpm", 
                       (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            y_pos += 35
            face_color = (0, 255, 0) if self.face_detected else (255, 255, 0)
            cv2.putText(image, f"Faces: {self.face_count}", 
                       (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, face_color, 2)
            
            y_pos += 25
            if self.person_detected:
                distance_color = (0, 255, 0) if not self.distance_warning else (0, 0, 255)
                cv2.putText(image, f"Distance: {self.current_distance:.2f}m", 
                           (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, distance_color, 2)
            
            return image
            
        except Exception as e:
            return image
    
    def generate_mjpeg_stream(self, stream_type='face'):
        """生成MJPEG流"""
        while True:
            try:
                with self.frame_lock:
                    if stream_type == 'face':
                        frame = self.latest_face_frame
                    elif stream_type == 'color':
                        frame = self.latest_color_frame
                    else:
                        frame = self.latest_depth_frame
                    
                    if frame is None:
                        frame = np.zeros((480, 640, 3), dtype=np.uint8)
                        cv2.putText(frame, f"No camera data", 
                                   (200, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
                frame = cv2.resize(frame, (640, 480))
                
                encode_params = [cv2.IMWRITE_JPEG_QUALITY, 85]
                ret, buffer = cv2.imencode('.jpg', frame, encode_params)
                
                if ret:
                    frame_bytes = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                
                time.sleep(0.033)  # 30fps
                
            except Exception as e:
                self.error_print("MJPEG流错误", e)
                time.sleep(0.1)
    
    def process_radar_data(self, raw_data):
        """处理雷达数据"""
        try:
            byte_array = np.frombuffer(raw_data, dtype=np.uint8)
            phase_data = byte_array * (2 * np.pi / 255.0)
            phase_unwrapped = np.unwrap(phase_data)
            
            chest_region = phase_unwrapped[100:156]
            phase_mean = np.mean(chest_region)
            
            # 检查相位数据的合理性
            if np.isnan(phase_mean) or np.isinf(phase_mean):
                self.error_print("相位数据包含异常值")
                return None
            
            current_time = time.time()
            self.phase_history.append(phase_mean)
            self.time_history.append(current_time)
            
            vital_signs = self.detect_vital_signs_with_stable_filtering()
            
            return {
                'phase_unwrapped': phase_unwrapped.tolist(),
                'phase_mean': float(phase_mean),
                'breathing_rate': float(self.breathing_rate),
                'heart_rate': float(self.heart_rate),
                'breathing_quality': float(self.breathing_quality),
                'heart_quality': float(self.heart_quality),
                'vital_signs': vital_signs
            }
            
        except Exception as e:
            self.error_print("数据处理错误", e)
            return None
    
    def connect_serial(self):
        """连接串口"""
        try:
            self.serial_conn = serial.Serial(
                port=self.serial_port,
                baudrate=self.baudrate,
                timeout=0.1
            )
            self.success_print(f"串口连接成功: {self.serial_port}")
            return True
        except Exception as e:
            self.error_print(f"串口连接失败", e)
            return False
    
    def read_radar_frame(self):
        """读取雷达帧"""
        if not self.serial_conn:
            return None
        try:
            data = self.serial_conn.read(self.data_length + 2)
            if len(data) >= self.data_length:
                return data[-self.data_length:]
            return None
        except:
            return None
    
    def radar_reader_thread(self):
        """雷达读取线程"""
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
        """启动雷达"""
        if not self.connect_serial():
            return False
        
        self.is_running = True
        self.start_time = time.time()
        
        self.radar_thread = threading.Thread(target=self.radar_reader_thread)
        self.radar_thread.daemon = True
        self.radar_thread.start()
        
        self.success_print("雷达系统已启动")
        return True
    
    def start_camera(self):
        """启动摄像头"""
        if not self.init_camera_with_face_detection():
            return False
        
        self.camera_running = True
        
        self.camera_thread = threading.Thread(target=self.camera_stream_thread)
        self.camera_thread.daemon = True
        self.camera_thread.start()
        
        self.success_print("摄像头系统已启动")
        return True
    
    def stop_radar(self):
        """停止雷达"""
        self.is_running = False
        
        # 重置数据
        self.breathing_rate = 0.0
        self.heart_rate = 0.0
        self.breathing_amplitude = 0.0
        self.heart_amplitude = 0.0
        self.breathing_quality = 0.0
        self.heart_quality = 0.0
        self.breathing_signal = []
        self.heart_signal = []
        
        self.phase_history.clear()
        self.time_history.clear()
        
        with self.data_lock:
            self.latest_phase_data = None
        
        self.frame_count = 0
        
        if self.serial_conn:
            self.serial_conn.close()
            self.serial_conn = None
        
        self.success_print("雷达已停止")
    
    def stop_camera(self):
        """停止摄像头"""
        self.camera_running = False
        
        with self.frame_lock:
            self.latest_color_frame = None
            self.latest_depth_frame = None
            self.latest_face_frame = None
        
        with self.face_lock:
            self.face_count = 0
            self.face_detected = False
            self.face_positions = []
            self.person_detected = False
            self.distance_warning = False
            self.current_distance = 0.0
        
        if self.camera_pipeline:
            self.camera_pipeline.stop()
            self.camera_pipeline = None
        
        self.success_print("摄像头已停止")
    
    def get_latest_data(self):
        """获取最新数据"""
        current_time = time.time()
        
        fps = 0
        if self.start_time and self.is_running:
            elapsed = current_time - self.start_time
            fps = self.frame_count / elapsed if elapsed > 0 else 0
        
        with self.data_lock:
            current_phase = self.latest_phase_data
        
        with self.face_lock:
            current_distance = self.current_distance
            person_detected = self.person_detected
            face_detected = self.face_detected
            face_count = self.face_count
            distance_warning = self.distance_warning
        
        if not self.is_running:
            return {
                'frame_count': 0,
                'fps': 0.0,
                'current_phase': None,
                'is_running': False,
                'camera_running': bool(self.camera_running),
                'breathing_rate': 0.0,
                'heart_rate': 0.0,
                'breathing_quality': 0.0,
                'heart_quality': 0.0,
                'breathing_signal': [],
                'heart_signal': [],
                'distance': 0.0,
                'person_detected': False,
                'face_detected': False,
                'face_count': 0,
                'distance_warning': False
            }
        
        return {
            'frame_count': int(self.frame_count),
            'fps': float(round(fps, 1)),
            'current_phase': current_phase,
            'is_running': bool(self.is_running),
            'camera_running': bool(self.camera_running),
            'breathing_rate': float(round(self.breathing_rate, 1)),
            'heart_rate': float(round(self.heart_rate, 1)),
            'breathing_quality': float(round(self.breathing_quality, 3)),
            'heart_quality': float(round(self.heart_quality, 3)),
            'breathing_signal': self.breathing_signal[-100:] if self.breathing_signal else [],
            'heart_signal': self.heart_signal[-100:] if self.heart_signal else [],
            'distance': float(round(current_distance, 2)),
            'person_detected': bool(person_detected),
            'face_detected': bool(face_detected),
            'face_count': int(face_count),
            'distance_warning': bool(distance_warning)
        }
    
    def debug_print(self, message):
        timestamp = datetime.now().strftime('%H:%M:%S')
        print(f"[{timestamp}] DEBUG: {message}")
    
    def error_print(self, message, exception=None):
        timestamp = datetime.now().strftime('%H:%M:%S')
        print(f"[{timestamp}] ERROR: {message}")
        if exception:
            print(f"Exception: {str(exception)}")
        self.error_count += 1
    
    def success_print(self, message):
        timestamp = datetime.now().strftime('%H:%M:%S')
        print(f"[{timestamp}] SUCCESS: {message}")
        self.success_count += 1

# 创建呼吸心跳检测系统实例
breathing_system = BreathingHeartSystem()

# Flask路由
@app.route('/')
def index():
    return """
<!DOCTYPE html>
<html>
<head>
    <title>呼吸心跳检测系统</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/3.9.1/chart.min.js"></script>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f0f0f0; }
        .container { max-width: 1800px; margin: 0 auto; }
        .header { text-align: center; margin-bottom: 30px; }
        .controls { text-align: center; margin: 20px 0; }
        .status { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 20px 0; }
        .status-card { padding: 20px; background: white; border-radius: 10px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); text-align: center; }
        .video-section { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 20px; margin: 20px 0; }
        .video-card { background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }
        .charts { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 20px; margin: 20px 0; }
        .chart-container { background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }
        .chart-wrapper { height: 300px; }
        .video-stream { width: 100%; height: 300px; border: 2px solid #ddd; border-radius: 5px; }
        button { padding: 12px 24px; margin: 5px; border: none; border-radius: 5px; cursor: pointer; font-size: 16px; }
        .btn-start { background: #28a745; color: white; }
        .btn-stop { background: #dc3545; color: white; }
        .status-ok { color: #28a745; }
        .status-error { color: #dc3545; }
        .status-warning { color: #ffc107; }
        .face-indicator { display: inline-block; width: 12px; height: 12px; border-radius: 50%; margin-left: 8px; }
        .face-detected { background-color: #28a745; }
        .face-not-detected { background-color: #dc3545; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🫁 呼吸心跳检测系统</h1>
            <p>基于24GHz毫米波雷达的非接触式生命体征监测</p>
        </div>
        
        <div class="controls">
            <button class="btn-start" onclick="startRadar()">启动雷达</button>
            <button class="btn-start" onclick="startCamera()">启动摄像头</button>
            <button class="btn-stop" onclick="stopRadar()">停止雷达</button>
            <button class="btn-stop" onclick="stopCamera()">停止摄像头</button>
        </div>
        
        <div class="status">
            <div class="status-card">
                <h3>🎯 雷达状态</h3>
                <div id="radarStatus" class="status-error">未连接</div>
                <div>帧数: <span id="frameCount">0</span></div>
                <div>FPS: <span id="fps">0</span></div>
            </div>
            
            <div class="status-card">
                <h3>👤 人脸检测</h3>
                <div id="faceStatus" class="status-warning">未检测</div>
                <div>人脸数: <span id="faceCount">0</span> <span class="face-indicator face-not-detected" id="faceIndicator"></span></div>
                <div>距离: <span id="distance">--</span></div>
            </div>
            
            <div class="status-card">
                <h3>🫁 呼吸检测</h3>
                <div id="breathingRate" class="status-warning">-- 次/分</div>
                <div>质量: <span id="breathingQuality">--</span></div>
            </div>
            
            <div class="status-card">
                <h3>❤️ 心率检测</h3>
                <div id="heartRate" class="status-warning">-- 次/分</div>
                <div>质量: <span id="heartQuality">--</span></div>
            </div>
            
            <div class="status-card">
                <h3>📹 摄像头</h3>
                <div id="cameraStatus" class="status-error">未连接</div>
                <div>状态: <span id="cameraRunning">停止</span></div>
            </div>
        </div>
        
        <div class="video-section">
            <div class="video-card">
                <h3>👤 人脸检测视频流</h3>
                <img id="faceStream" src="" class="video-stream" alt="人脸检测视频流">
            </div>
            <div class="video-card">
                <h3>📹 彩色视频流</h3>
                <img id="colorStream" src="" class="video-stream" alt="彩色视频流">
            </div>
            <div class="video-card">
                <h3>🌈 深度视频流</h3>
                <img id="depthStream" src="" class="video-stream" alt="深度视频流">
            </div>
        </div>
        
        <div class="charts">
            <div class="chart-container">
                <h3>雷达相位信号</h3>
                <div class="chart-wrapper">
                    <canvas id="radarChart"></canvas>
                </div>
            </div>
            
            <div class="chart-container">
                <h3>呼吸波形 📈</h3>
                <div class="chart-wrapper">
                    <canvas id="breathingChart"></canvas>
                </div>
            </div>
            
            <div class="chart-container">
                <h3>心率波形 ❤️</h3>
                <div class="chart-wrapper">
                    <canvas id="heartChart"></canvas>
                </div>
            </div>
        </div>
    </div>

    <script>
        let radarChart, breathingChart, heartChart;
        let updateInterval;
        
        function initCharts() {
            const radarCtx = document.getElementById('radarChart').getContext('2d');
            radarChart = new Chart(radarCtx, {
                type: 'line',
                data: {
                    labels: Array.from({length: 256}, (_, i) => i + 1),
                    datasets: [{
                        label: '相位信号',
                        data: new Array(256).fill(0),
                        borderColor: '#007bff',
                        borderWidth: 2,
                        pointRadius: 0
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    animation: false,
                    scales: {
                        y: {
                            title: {
                                display: true,
                                text: '相位 (弧度)'
                            }
                        }
                    }
                }
            });
            
            const breathingCtx = document.getElementById('breathingChart').getContext('2d');
            breathingChart = new Chart(breathingCtx, {
                type: 'line',
                data: {
                    labels: Array.from({length: 100}, (_, i) => i + 1),
                    datasets: [{
                        label: '呼吸信号',
                        data: new Array(100).fill(0),
                        borderColor: '#28a745',
                        borderWidth: 2,
                        pointRadius: 0
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    animation: false,
                    scales: {
                        y: {
                            title: {
                                display: true,
                                text: '幅度 (弧度)'
                            }
                        }
                    }
                }
            });
            
            const heartCtx = document.getElementById('heartChart').getContext('2d');
            heartChart = new Chart(heartCtx, {
                type: 'line',
                data: {
                    labels: Array.from({length: 100}, (_, i) => i + 1),
                    datasets: [{
                        label: '心率信号',
                        data: new Array(100).fill(0),
                        borderColor: '#dc3545',
                        borderWidth: 2,
                        pointRadius: 0
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    animation: false,
                    scales: {
                        y: {
                            title: {
                                display: true,
                                text: '幅度 (弧度)'
                            }
                        }
                    }
                }
            });
        }
        
        function updateData() {
            fetch('/api/data')
                .then(response => response.json())
                .then(data => {
                    // 更新基本状态
                    document.getElementById('radarStatus').textContent = data.is_running ? '运行中' : '未连接';
                    document.getElementById('radarStatus').className = data.is_running ? 'status-ok' : 'status-error';
                    
                    document.getElementById('cameraStatus').textContent = data.camera_running ? '运行中' : '未连接';
                    document.getElementById('cameraStatus').className = data.camera_running ? 'status-ok' : 'status-error';
                    document.getElementById('cameraRunning').textContent = data.camera_running ? '运行' : '停止';
                    
                    // 更新人脸检测状态
                    document.getElementById('faceStatus').textContent = data.face_detected ? '已检测' : '未检测';
                    document.getElementById('faceStatus').className = data.face_detected ? 'status-ok' : 'status-warning';
                    document.getElementById('faceCount').textContent = data.face_count;
                    
                    const faceIndicator = document.getElementById('faceIndicator');
                    faceIndicator.className = data.face_detected ? 'face-indicator face-detected' : 'face-indicator face-not-detected';
                    
                    // 更新生命体征
                    document.getElementById('breathingRate').textContent = data.breathing_rate > 0 ? data.breathing_rate + ' 次/分' : '-- 次/分';
                    document.getElementById('heartRate').textContent = data.heart_rate > 0 ? data.heart_rate + ' 次/分' : '-- 次/分';
                    
                    document.getElementById('breathingQuality').textContent = data.breathing_quality.toFixed(2);
                    document.getElementById('heartQuality').textContent = data.heart_quality.toFixed(2);
                    
                    document.getElementById('fps').textContent = data.fps;
                    document.getElementById('frameCount').textContent = data.frame_count;
                    
                    // 更新距离
                    const distanceEl = document.getElementById('distance');
                    if (data.person_detected) {
                        distanceEl.textContent = data.distance + 'm';
                        distanceEl.style.color = data.distance_warning ? '#dc3545' : '#28a745';
                    } else {
                        distanceEl.textContent = '--';
                        distanceEl.style.color = '#ffc107';
                    }
                    
                    // 更新图表
                    if (data.current_phase && data.current_phase.phase_unwrapped) {
                        radarChart.data.datasets[0].data = data.current_phase.phase_unwrapped;
                        radarChart.update('none');
                    }
                    
                    if (data.breathing_signal && data.breathing_signal.length > 0) {
                        breathingChart.data.datasets[0].data = data.breathing_signal;
                        breathingChart.update('none');
                    }
                    
                    if (data.heart_signal && data.heart_signal.length > 0) {
                        heartChart.data.datasets[0].data = data.heart_signal;
                        heartChart.update('none');
                    }
                })
                .catch(error => {
                    console.error('数据获取错误:', error);
                });
        }
        
        function startRadar() {
            fetch('/api/start-radar', { method: 'POST' })
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        alert('雷达启动成功');
                        if (!updateInterval) {
                            updateInterval = setInterval(updateData, 200);
                        }
                    } else {
                        alert('雷达启动失败: ' + data.message);
                    }
                });
        }
        
        function startCamera() {
            fetch('/api/start-camera', { method: 'POST' })
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        alert('摄像头启动成功');
                        document.getElementById('faceStream').src = '/video_face';
                        document.getElementById('colorStream').src = '/video_color';
                        document.getElementById('depthStream').src = '/video_depth';
                    } else {
                        alert('摄像头启动失败: ' + data.message);
                    }
                });
        }
        
        function stopRadar() {
            fetch('/api/stop-radar', { method: 'POST' })
                .then(response => response.json())
                .then(data => {
                    alert('雷达已停止');
                    radarChart.data.datasets[0].data = new Array(256).fill(0);
                    breathingChart.data.datasets[0].data = new Array(100).fill(0);
                    heartChart.data.datasets[0].data = new Array(100).fill(0);
                    radarChart.update();
                    breathingChart.update();
                    heartChart.update();
                });
        }
        
        function stopCamera() {
            fetch('/api/stop-camera', { method: 'POST' })
                .then(response => response.json())
                .then(data => {
                    alert('摄像头已停止');
                    document.getElementById('faceStream').src = '';
                    document.getElementById('colorStream').src = '';
                    document.getElementById('depthStream').src = '';
                });
        }
        
        document.addEventListener('DOMContentLoaded', function() {
            initCharts();
            updateInterval = setInterval(updateData, 200);
        });
    </script>
</body>
</html>
    """

@app.route('/video_face')
def video_face():
    """人脸检测视频流"""
    return Response(breathing_system.generate_mjpeg_stream('face'),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_color')
def video_color():
    """彩色视频流"""
    return Response(breathing_system.generate_mjpeg_stream('color'),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_depth')
def video_depth():
    """深度视频流"""
    return Response(breathing_system.generate_mjpeg_stream('depth'),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/start-radar', methods=['POST'])
def start_radar():
    try:
        success = breathing_system.start_radar()
        return jsonify({'success': success, 'message': '雷达已启动' if success else '启动失败'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/stop-radar', methods=['POST'])
def stop_radar():
    try:
        breathing_system.stop_radar()
        return jsonify({'success': True, 'message': '雷达已停止'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/start-camera', methods=['POST'])
def start_camera():
    try:
        success = breathing_system.start_camera()
        return jsonify({'success': success, 'message': '摄像头已启动' if success else '启动失败'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/stop-camera', methods=['POST'])
def stop_camera():
    try:
        breathing_system.stop_camera()
        return jsonify({'success': True, 'message': '摄像头已停止'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/data')
def get_data():
    try:
        data = breathing_system.get_latest_data()
        return jsonify(data)
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    print("🫁 启动呼吸心跳检测系统")
    print("=" * 60)
    print("基于24GHz毫米波雷达的非接触式生命体征监测")
    print("🌐 访问地址示例: http://10.162.133.43:5000（按实际 IP 修改）")
    print("按 Ctrl+C 停止服务")
    
    try:
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
    except KeyboardInterrupt:
        print("\n正在关闭服务器...")
        breathing_system.stop_radar()
        breathing_system.stop_camera()
        print("服务器已关闭")