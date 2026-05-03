import os

WIDTH = 640
HEIGHT = 480
FPS = 30

# 树莓派上 shumeipai.py 的 HTTP 根地址，例如 http://10.162.133.43:5000
RADAR_PI_BASE = os.environ.get("RADAR_PI_BASE", "").rstrip("/")

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

# ---------- Web 人脸：facenet-pytorch ----------
FACE_PROFILE_FILE = "face_profiles_v2.npz"
FACE_PROFILE_VERSION = 2
FACE_MAX_PROFILES = 10

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
