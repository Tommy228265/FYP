import os

WIDTH = 640
HEIGHT = 480
FPS = 30

# ---------- Web 视频流：减轻卡顿、简化画面 ----------
# 仅绘制绿色人脸框，不绘制顶部状态栏、底部识别横幅、人名/深度等叠字（相机自带 OSD 需在硬件菜单关闭）
# 恢复完整 HUD：VIDEO_MINIMAL_OVERLAY=0
VIDEO_MINIMAL_OVERLAY = os.environ.get("VIDEO_MINIMAL_OVERLAY", "1").strip().lower() not in (
    "0",
    "false",
    "no",
)
# MJPEG JPEG 质量（越低编码越快、带宽越小；Wi‑Fi 可试 70–80）
VIDEO_JPEG_QUALITY = max(40, min(95, int(os.environ.get("VIDEO_JPEG_QUALITY", "78"))))
VIDEO_JPEG_DEPTH_QUALITY = max(40, min(95, int(os.environ.get("VIDEO_JPEG_DEPTH_QUALITY", "72"))))
# 设为 0 则不再每帧编码深度 MJPEG（不看 /video_depth 时可减负）
VIDEO_ENCODE_DEPTH_STREAM = os.environ.get("VIDEO_ENCODE_DEPTH_STREAM", "1").strip().lower() not in (
    "0",
    "false",
    "no",
)

# 树莓派 shumeipai.py 的 HTTP 根（雷达 JSON + USB 摄像头 MJPEG）。默认固定为你的局域网树莓派；可用环境变量覆盖。
_DEFAULT_RADAR_PI_BASE = "http://10.245.232.43:5000"
RADAR_PI_BASE = (os.environ.get("RADAR_PI_BASE") or _DEFAULT_RADAR_PI_BASE).strip().rstrip(
    "/"
)

# 默认 True：视频从树莓派拉流，本机不占用 Intel RealSense。若临时改用本机 D435，请设 USE_PI_CAMERA=0。
_USE_PI_RAW = (os.environ.get("USE_PI_CAMERA") or "1").strip().lower()
USE_PI_CAMERA = _USE_PI_RAW not in ("0", "false", "no")

WINDOW_NAME = "D435 Person Detection"
DEPTH_WINDOW_NAME = "D435 Depth"

# YOLO 人体检测参数
YOLO_MODEL = "yolov8n.pt"
YOLO_CONFIDENCE = 0.35
YOLO_IMGSZ = 640
YOLO_DEVICE = "cpu"

# 过滤过小的框，避免远处噪声框
MIN_BBOX_WIDTH = 40
MIN_BBOX_HEIGHT = 80

# 简单多帧追踪参数（减少单帧闪烁）
TRACK_IOU_THRESHOLD = 0.3
TRACK_MIN_HITS = 1
TRACK_MAX_MISSES = 8
TRACK_DRAW_MAX_MISSES = 3

# 深度 ROI 有效范围（米），用于中值统计
DEPTH_VALID_MIN_M = 0.15
DEPTH_VALID_MAX_M = 10.0
DEPTH_MIN_VALID_PIXELS = 5

# 生命体征推荐站位（米）：用于 UI 提示与 depth_zone
DEPTH_VITAL_OPTIMAL_MIN_M = 0.45
DEPTH_VITAL_OPTIMAL_MAX_M = 1.9

# 深度与雷达多距离门匹配的粗略场景范围（米）：近 -> 低 bin 通道，远 -> 高 bin 通道
FUSION_DEPTH_SCENE_MIN_M = float(os.environ.get("FUSION_DEPTH_SCENE_MIN_M", "0.35"))
FUSION_DEPTH_SCENE_MAX_M = float(os.environ.get("FUSION_DEPTH_SCENE_MAX_M", "2.3"))
# 设 0 / false 则退回「深度排序名次 ↔ 通道名次」配对
FUSION_USE_DEPTH_BIN_MATCH = os.environ.get(
    "FUSION_USE_DEPTH_BIN_MATCH", "1"
).strip() not in ("0", "false", "False")

# 深度可视化缩放（仅用于显示）
DEPTH_VIS_ALPHA = 0.03

# 人脸质量：有有效 ROI 深度（米）时，要求落在生命体征推荐带内才视为高质（录入/识别）；无深度读数时不加此限制
FACE_DEPTH_QUALITY_GATE = os.environ.get("FACE_DEPTH_QUALITY_GATE", "1").strip().lower() not in (
    "0",
    "false",
    "no",
)

# 识别稳定锁定：滑动窗口内同一组人名占比超过阈值则冻结 API/雷达图例（仅保留绿框与视频）
RECOGNITION_LOCK_WINDOW_SEC = float(os.environ.get("RECOGNITION_LOCK_WINDOW_SEC", "5"))
RECOGNITION_LOCK_MAJORITY = float(os.environ.get("RECOGNITION_LOCK_MAJORITY", "0.51"))
RECOGNITION_LOCK_MIN_VOTES = max(3, int(os.environ.get("RECOGNITION_LOCK_MIN_VOTES", "10")))

# 年龄：参考年龄存在且某次预测判定正确后，该 track 的展示年龄冻结到该值，直至停止识别/重新开识别等
AGE_LOCK_ON_CORRECT = os.environ.get("AGE_LOCK_ON_CORRECT", "1").strip().lower() not in (
    "0",
    "false",
    "no",
)

# ---------- Web 人脸：facenet-pytorch ----------
FACE_PROFILE_FILE = "face_profiles_v2.npz"
FACE_PROFILE_META_FILE = "face_profile_meta.json"
FACE_PROFILE_VERSION = 2
FACE_MAX_PROFILES = 10
DISPLAY_NAME_MAX_LEN = 40

ENROLL_SAMPLES_TARGET = 20
ENROLL_SAMPLE_INTERVAL_SEC = 0.25

# 余弦相似度阈值（归一化嵌入）
FACE_RECOGNITION_THRESHOLD = 0.42

# 录入完成后与已有人脸模板比较；高于该值判定为重复面容
FACE_DUPLICATE_THRESHOLD = 0.72

# MTCNN 人脸置信度下限
FACE_MIN_MTCNN_PROB = 0.85

# 拉普拉斯清晰度下限；0 表示不启用
FACE_MIN_BLUR_VARIANCE = 40.0

# MTCNN 最小人脸边长（像素）
FACE_MTCNN_MIN_FACE_SIZE = 60

# InceptionResnet 预训练权重名
FACE_PRETRAINED = "vggface2"

# ---------- 年龄段估计：EfficientNet-B0 迁移学习 ----------
AGE_MODEL_FILE = "age_model_effnet_b0.pth"
AGE_INPUT_SIZE = 224
AGE_CALIBRATION_SHIFT = 1
AGE_SMOOTHING_WINDOW = 7
AGE_TRAIN_UTKFACE_ROOT = "UTKFace"
AGE_TRAIN_FAIRFACE_ROOT = None
AGE_TRAIN_FAIRFACE_CSV = None
# PhysFormer 视觉心率（face_app 内懒加载，需 GPU + physformer/PhysFormer 代码）
# 设 PHYSFORMER_ENABLED=0 可关闭；PHYSFORMER_WEIGHTS 指向 .pkl，留空则自动在 weights/ 与微调目录中查找
PHYSFORMER_ENABLED = os.environ.get("PHYSFORMER_ENABLED", "1").strip() not in (
    "0",
    "false",
    "False",
)
PHYSFORMER_WEIGHTS = os.environ.get("PHYSFORMER_WEIGHTS", "").strip()

AGE_CLASSES = [
    {"id": "child", "label": "儿童", "label_en": "Child", "range": "0-12"},
    {"id": "teen", "label": "青少年", "label_en": "Teen", "range": "13-19"},
    {"id": "young_adult", "label": "青年", "label_en": "Young Adult", "range": "20-35"},
    {"id": "middle_aged", "label": "中年", "label_en": "Middle Age", "range": "36-55"},
    {"id": "senior", "label": "老年", "label_en": "Senior", "range": "56+"},
]
