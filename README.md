# FYP：Intel RealSense D435 人体检测与多人脸识别

本项目基于 **Intel RealSense D435** 彩色与深度流（亦可切换为树莓派 USB 摄像头推流），结合 **YOLOv8** 做人体检测、**深度 ROI 中值** 估计距离（树莓派 RGB 模式下无深度图），并通过 **Web 页面** 完成最多 **10 个面容档案** 的录入、删除、重复面容检测与身份识别。配置集中在 `config.py`，便于实验与论文中复现实验参数。

---

## 树莓派采集端与上位机分工（可选）

| 组件 | 运行位置 | 说明 |
|------|----------|------|
| 摄像头（UVC 或 RealSense D435） | 树莓派 | `shumeipai.py` 默认 `FYP_CAMERA_MODE=auto` 优先 RealSense（与 `jianjie.py` 一致）；可用 `FYP_CAMERA_ENABLE=0` 关闭；MJPEG：**`/camera/rgb`**（彩色）、**`/camera/depth`**（深度伪彩，仅 depth+color 启动成功时；否则该端点为占位图） |
| 毫米波雷达串口 | 树莓派 | 同上脚本解析相位并通过 `/api/radar` 提供 JSON |
| 人脸识别、年龄段、PhysFormer、档案存储 | **上位机** `face_app.py` | 浏览器始终访问上位机端口（默认 `:5000`） |
| 视频构图拉流 | 上位机 | 设置 **`RADAR_PI_BASE=http://10.245.232.43:5000`** 且 **`USE_PI_CAMERA=1`** 后启动 `face_app.py`（换路由器时请改 IP）；上位机从树莓派 MJPEG 解码并在本机跑算法，本地不再占用 Intel RealSense |

**上位机一键示例（Windows PowerShell）：**

```powershell
$env:RADAR_PI_BASE="http://10.245.232.43:5000"; $env:USE_PI_CAMERA="1"; python face_app.py
```

**依赖**：树莓派需安装 `opencv-python`、`Flask` 等与现有 `shumeipai.py` 一致的依赖；若摄像头为 **Intel RealSense D435**，还需 **`pyrealsense2`**（与 `jianjie.py` 相同，不经 OpenCV 按索引打开 `/dev/video0`）。上位机可选安装 `markdown` 以便 `/readme` 页面渲染表格。

**RealSense 与树莓派**：在 ARM 上执行 `pip install pyrealsense2` 经常出现 *No matching distribution*，这是正常现象（PyPI 未必提供当前 Python/架构的 wheel）。请按 Intel 官方文档从源码编译 **librealsense** 并生成 Python 绑定（例如文档中的 Raspberry Pi / ARM 安装流程：<https://github.com/IntelRealSense/librealsense/blob/master/doc/installation_raspbian.md>）。若日志出现 **`No device connected`**，说明 Python 库已能加载，但 **USB 侧未枚举到相机**：换 **USB3** 口、换数据线、保证供电，并在 Pi 上运行 `rs-enumerate-devices`（或 `lsusb`）确认设备出现。RealSense 失败回退到 V4L2 时，可用 `ls /dev/video*` 查看节点，并通过环境变量 **`FYP_CAMERA_DEVICE=/dev/videoN`** 指定实际彩色节点（有时不是 `video0`）。

---

## 功能概览

| 模块 | 作用 |
|------|------|
| `test.py` | 本地 OpenCV 窗口：YOLO 检测行人、简单 IoU 追踪、人体框内深度中值、彩色/深度可视化 |
| `face_app.py` + `templates/index.html` | Flask Web：视频流、最多 10 个面容录入/删除、重复检测、识别模式、显示相似度、深度与年龄段 |
| `face_identity.py` | MTCNN 人脸检测与对齐 + InceptionResnetV1 提取 512 维嵌入（Facenet 路线） |
| `age_estimator.py` | EfficientNet-B0 年龄段分类推理（加载训练好的本地权重） |
| `train_age_model.py` | 基于 UTKFace/FairFace 训练年龄段分类模型 |
| `realsense_utils.py` | 深度缩放与 ROI 内有效深度中值（米） |
| `config.py` | 分辨率、YOLO、追踪、深度与人脸相关超参数 |


---

## 模型、训练数据与文献总览（论文撰写用）

下表汇总本仓库中 **曾使用或引用的所有可学习模型**、**训练/预训练数据**、**代表论文与工程来源**，以及 **本机训练脚本的超参数出处**（见各 `argparse` 默认值与 `config.py`）。**非学习模块**（纯信号处理、简单 FFT）亦单独说明，避免与深度学习混为一谈。

### 总览表

| 模块 | 模型/方法 | 主要用途 | 预训练或数据 | 代表文献 / 工程 |
|------|-----------|----------|--------------|-----------------|
| 人体检测 | YOLOv8-n（Ultralytics） | `test.py` 中检测 person；COCO 预训练权重 | COCO 2017 检测类 | 工程： [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)；方法线：Redmon 等 YOLO 系列；v8 见 Ultralytics 文档 |
| 人脸检测与对齐 | MTCNN（facenet-pytorch） | 人脸框与对齐，阈值等见 `face_identity.py` | WIDER Face 等公开流程使用的开源实现 | Zhang et al., *Joint Face Detection and Alignment using Multi-task Cascaded Convolutional Networks*, IEEE Signal Processing Letters, 2016 |
| 人脸嵌入 | InceptionResnetV1，`pretrained='vggface2'` | 512 维 L2 归一化嵌入，识别与建档 | **VGGFace2** 预训练（经 facenet-pytorch 分发） | 嵌入框架：Schroff et al., *FaceNet: A Unified Embedding for Face Recognition and Clustering*, CVPR, 2015；架构：Szegedy et al., *Inception-v4, Inception-ResNet*, AAAI, 2017；数据：Cao et al., *VGGFace2: A Dataset for Recognising Faces across Pose and Age*, FG, 2018 |
| 年龄段 | EfficientNet-B0 分类头（5 类） | `age_estimator.py` 推理；权重来自 `train_age_model.py` | **ImageNet** 预训练骨干（torchvision）+ **UTKFace** 或 **FairFace** 微调 | 骨干：Tan & Le, *EfficientNet*, ICML, 2019；UTKFace：Zhang et al., *Age Progression/Regression by Conditional Adversarial Autoencoder*, CVPR, 2017；FairFace：Kärkkäinen & Joo, *FairFace: Face Attribute Dataset for Balanced Race, Gender, and Age*, arXiv/WACV 相关引用以数据集页面为准 |
| 远程心率 | PhysFormer（`ViT_ST_ST_Compact3_TDC_gra_sharp`） | `physformer_engine.py`；160×128×128 时序块 | **VIPL-HR 上 fold1 官方权重** 初始化 + **UBFC-rPPG** 本机微调 | Yu et al., *PhysFormer: Facial Video-based Physiological Measurement with Temporal Difference Transformer*, CVPR, 2022；预训练数据与 VIPL：Niu et al., *RhythmNet: End-to-End Heart Rate Estimation from Face*, IEEE T-IP, 2019（与官方 checkpoint 说明一致时引用）；微调数据：Bobbia et al., *Unsupervised skin tissue segmentation for remote photoplethysmography*, Pattern Recognition Letters, 2017 |
| 视觉呼吸 | 无训练模型 | ROI 绿色通道均值 + 短时 FFT 峰值（`visual_respiration.py`） | 无 | 工程启发来自远程 PPG/成像式生理测量综述类文献即可，勿宣称临床级 |
| 毫米波雷达 | 无深度学习 | `shumeipai.py`：相位解缠、带通、FFT/自相关、EMA | 无 | FMCW 生命体征经典信号处理；引用雷达非接触生理监测综述或教材即可 |

---

### 1. 人体检测：YOLOv8-n

- **实现**：`ultralytics` 包，`config.YOLO_MODEL` 默认为 `yolov8n.pt`，检测类别在 `test.py` 中限制为 **person（class 0）**。
- **权重来源**：首次运行由 Ultralytics **自动下载**；在 **MS COCO** 目标检测任务上预训练（80 类中的 `person`）。
- **本项目推理参数（来自 `config.py`）**：`YOLO_CONFIDENCE=0.35`，`YOLO_IMGSZ=640`，`YOLO_DEVICE` 默认 `cpu`（可按机器改为 `0` 使用 GPU）。
- **文献与引用**：工程文档与许可见 [Ultralytics](https://github.com/ultralytics/ultralytics)；论文中可引用 YOLO 系列原始工作（如 Redmon et al., YOLO；后续版本按课程要求选引）。

---

### 2. 人脸：MTCNN + InceptionResnetV1（facenet-pytorch）

- **实现**：`face_identity.py`，依赖 **`facenet_pytorch`** 的 `MTCNN` 与 `InceptionResnetV1`。
- **MTCNN**：三级级联检测与五点对齐；默认 `thresholds=[0.6, 0.7, 0.7]`，`factor=0.709`，`image_size=160`，`margin=14`，与常见开源实现一致。
- **InceptionResnetV1**：`pretrained='vggface2'`（`config.FACE_PRETRAINED`），输出 **512 维** 嵌入，代码内 **L2 归一化** 后做余弦相似度。
- **数据与权重**：**VGGFace2** 大规模人脸识别数据上训练的公开权重（通过 PyTorch 加载）；**非**在本项目中从零训练。
- **文献**：FaceNet 嵌入学习框架（Schroff et al., CVPR 2015）；Inception-ResNet 结构（Szegedy et al., AAAI 2017）；VGGFace2 数据集（Cao et al., FG 2018）；MTCNN（Zhang et al., IEEE SPL 2016）。

---

### 3. 年龄段：EfficientNet-B0

- **结构**：`age_estimator.create_efficientnet_b0`：torchvision **EfficientNet-B0**，替换最后一层为 **5 类**（儿童 / 青少年 / 青年 / 中年 / 老年，年龄边界见 `config.AGE_CLASSES`）。
- **预训练**：训练脚本默认 **`ImageNet` 预训练骨干**（`torchvision.models.efficientnet_b0(weights=...)`）；可用 `--no-pretrained` 关闭对照实验。
- **微调数据（任选其一或组合）**：
  - **UTKFace**：文件名格式解析年龄 → 映射到 5 类（`train_age_model.py` 中 `UTKFaceDataset`）。
  - **FairFace**：CSV 中年龄/年龄段字段解析（`FairFaceDataset`）。
- **训练超参数（默认值来自 `train_age_model.py` 的 `argparse`）**：`epochs=12`，`warmup-epochs=2`（先冻结主干只训分类头），`batch-size=32`，`lr=3e-4`，`head-lr=8e-4`（warmup 阶段），`weight-decay=1e-4`，`label-smoothing=0.04`，`val-ratio=0.15`，`seed=42`；数据增强含 `RandomResizedCrop`、`ColorJitter`、`RandomErasing` 等（见脚本内 `train_transform`）。
- **推理**: `AGE_INPUT_SIZE=224`，归一化均值方差为 **ImageNet 标准** `[0.485,0.456,0.406]` / `[0.229,0.224,0.225]`。
- **文献**：EfficientNet（Tan & Le, ICML 2019）；UTKFace、FairFace 见上表；类别年龄段划分为本项目任务定义，非某一篇论文专有。

---

### 4. 远程心率：PhysFormer

- **网络**：`physformer/PhysFormer/model` 中 **`ViT_ST_ST_Compact3_TDC_gra_sharp`**，与仓库内训练脚本一致的超参实例：`image_size=(160,128,128)`，`patches=(4,4,4)`，`dim=96`，`ff_dim=144`，`num_heads=4`，`num_layers=12`，`dropout_rate=0.1`，`theta=0.7`（见 `physformer_engine.py` 与 `train_Physformer_160_UBFC.py`）。
- **预训练权重**：官方在 **VIPL-HR** 流程上得到的 **fold1** checkpoint（文件名常为 `Physformer_VIPL_fold1.pkl`），从 [ZitongYu/PhysFormer](https://github.com/ZitongYu/PhysFormer) README/Google Drive 获取；**非随机初始化**。
- **本机微调数据**：**UBFC-rPPG**，脚本 `train_Physformer_160_UBFC.py`；每个 `subject*` 含 `vid.avi` 与 `ground_truth.txt`。
- **训练默认超参（来自脚本 `argparse`）**：`lr=1e-4`，`batchsize=4`，优化器 **Adam**，`weight_decay=5e-5`，学习率调度 **StepLR**（`step_size=50`，`gamma=0.5`），`epochs=25`，`clip_stride=80`（滑窗起始间隔），`val_ratio=0.2`，`split_seed=42`，`num_workers` 默认 `0`（Windows 建议）；每 epoch 保存 `Physformer_UBFC_{fold}_{epoch}.pkl`。
- **实时推理（`physformer_engine.py`）**：每轨迹缓冲 **`CLIP_FRAMES=160`** 帧，人脸块 **128×128**，滑动 **`SLIDE=80`**；模型输出后经归一化，心率可由 **频域峰**（约 **0.65–3.5 Hz**）换算 BPM（实现细节见源码）。Web 展示权重默认 **`weights/Physformer_UBFC_best.pkl`** 或通过 `PHYSFORMER_WEIGHTS` 指定。
- **文献**：PhysFormer（Yu et al., CVPR 2022）；PhysFormer++（Yu et al., IJCV 2023，若换用++权重）；VIPL-HR / RhythmNet（Niu et al., IEEE T-IP 2019）；UBFC-rPPG（Bobbia et al., PRL 2017）。

---

### 5. 视觉呼吸估计（无训练）

- **方法**：人脸 ROI **绿色通道** 时间序列 + **rFFT**，在约 **0.12–0.55 Hz** 带内取峰值得呼吸率（`visual_respiration.py`），并对输出做 **EMA** 平滑。
- **论文表述建议**：标注为 **简易启发式 / 辅助 modality**，与深度学习或雷达物理模型区分。

---

### 6. 毫米波雷达管线（树莓派，无深度学习）

- **内容**：相位解缠、带通滤波、谐波抑制、**Hann 窗 FFT + 抛物线细化**、**自相关** 与 **EMA**（`shumeipai.py`）；与摄像头融合逻辑见 `radar_fusion.py`。
- **文献**：可引用 FMCW 非接触生命体征监测综述或教材；无需绑定某一神经网络论文。

---

### 7. 参数来源小结（便于答辩）

| 参数类别 | 主要来自 |
|----------|----------|
| YOLO / 人脸 / 深度 Web 门控 | `config.py` 与本项目实验设定 |
| 年龄段训练默认超参 | `train_age_model.py` 中 `argparse` 默认值 |
| PhysFormer 训练默认超参 | `physformer/PhysFormer/train_Physformer_160_UBFC.py` 中 `argparse` 默认值；网络结构字段与官方仓库模型定义一致 |
| PhysFormer 推理缓冲长度 | `physformer_engine.py` 中 `CLIP_FRAMES`、`SLIDE` 等常量 |
| 雷达频段与采样 | `shumeipai.py` 内 `sample_rate`、呼吸/心率频带等 |

若论文需写「超参数是否网格搜索」，请按你实际实验补充；本 README 仅反映 **代码中的默认实现**。

---

## 环境与依赖

- **硬件**：Intel RealSense D435（USB3），安装 [Intel RealSense SDK 2.0](https://github.com/IntelRealSense/librealsense) 及对应 Python 包 `pyrealsense2`。
- **Python**：建议使用独立 Conda 环境（例如 `realsense`）。
- **主要依赖**（按需安装）：

```text
pyrealsense2
opencv-python
numpy
ultralytics          # YOLO
torch / torchvision  # ultralytics 与 facenet-pytorch 已依赖
facenet-pytorch      # 人脸检测+嵌入
flask                # Web 服务
```

首次运行 YOLO 会下载 `yolov8n.pt`（或你配置的模型）；首次运行 Facenet 会下载 **MTCNN** 与 **InceptionResnetV1（vggface2）** 预训练权重，请保持网络可用。

---

## 项目结构

```text
FYP/
├── config.py              # 全局配置
├── test.py                # 人体检测 + 深度（OpenCV）
├── face_app.py            # Web 人脸录入与识别
├── face_identity.py       # 人脸嵌入引擎
├── age_estimator.py       # 年龄段估计推理
├── train_age_model.py     # 年龄段模型训练脚本
├── realsense_utils.py     # 深度工具函数
├── templates/
│   └── index.html         # Web 前端页面
├── yolov8n.pt             # YOLO 权重（首次运行后生成或下载）
└── face_profiles_v2.npz   # 人脸特征档案（录入后生成）
└── age_model_effnet_b0.pth # 年龄段模型权重（训练后生成）
```

### PhysFormer 心率模型（简要）

- **数据**：本机可仅用 **部分** [UBFC-rPPG](https://sites.google.com/view/ybenezeth/ubfcrppg)（`datasets/UBFC-rPPG/subject*/`，含 `vid.avi` 与 `ground_truth.txt`）；论文需写明子集规模。
- **权重**：先用官方 **VIPL fold1** `Physformer_VIPL_fold1.pkl`（[PhysFormer 仓库](https://github.com/ZitongYu/PhysFormer)），再微调；推理默认 `weights/Physformer_UBFC_best.pkl` 或 `PHYSFORMER_WEIGHTS`。
- **完整文献、默认训练超参、网络字段**：见上文 **「模型、训练数据与文献总览」**；训练命令见下文 **PhysFormer 训练命令**。

---

## 实现细节

### 1. RealSense 数据流

- 彩色流与深度流均为 **640×480@30fps**，深度与彩色 **对齐到彩色**（`rs.align(rs.stream.color)`），保证检测框与深度图在同一像素坐标系下。
- 深度原始值为 `uint16`，通过 `depth_sensor.get_depth_scale()` 转为米（与 `pyrealsense2` 文档一致）。

### 2. `test.py`：人体检测与距离

- **检测**：Ultralytics YOLOv8，仅 **person 类（class 0）**，置信度等见 `config.py`。
- **追踪**：基于 IoU 的简易多目标关联，用于减轻帧间框抖动；`TRACK_*` 控制匹配阈值、丢失容忍与显示策略。
- **距离**：在人体框内对 **有效深度像素** 取 **中值**（`realsense_utils.median_depth_meters`），避免单点深度跳变；无效时显示 `depth n/a`。
- **显示**：彩色图叠加框与标签；深度图经颜色映射仅用于可视化。

### 3. `face_app.py`：多人脸 Web 系统

- **后端**：Flask；`/video_feed` 以 **MJPEG multipart** 推送 JPEG 帧；`/api/status` 的 JSON 字段 **`vitals`** 含 PhysFormer 每人 **心率与缓冲进度**（启用时）。
- **前端**：`templates/index.html` 调用 `POST /api/enroll/start`、`/api/recognize/start`、`/api/stop`，轮询 `GET /api/status` 显示录入状态。
- **流程**：
  1. **录入**：按间隔采集 `ENROLL_SAMPLES_TARGET` 帧通过质量门控的嵌入，对嵌入做 **均值** 后再 **L2 归一化** 作为该人模板。
  2. **重复检测**：新模板与已有人脸模板比较；若相似度高于 `FACE_DUPLICATE_THRESHOLD`，自动取消本次录入。
  3. **识别**：当前帧嵌入与所有已录入模板计算 **余弦相似度**，取较大者；若低于 `FACE_RECOGNITION_THRESHOLD` 则判为 Unknown。
- **质量门控**：MTCNN 人脸置信度 ≥ `FACE_MIN_MTCNN_PROB`；可选拉普拉斯方差 ≥ `FACE_MIN_BLUR_VARIANCE`（`0` 表示关闭模糊过滤）。
- **深度**：在人脸框内同样用 **深度中值** 显示距离（米）。

### 4. `face_identity.py`：人脸嵌入

- **检测与对齐**：`facenet_pytorch.MTCNN`（`keep_all=True`），在检测框中选取 **面积最大** 的人脸。
- **嵌入**：`InceptionResnetV1(pretrained='vggface2')`，输出 **512 维**，经 **L2 归一化** 后用于余弦相似度。
- **说明**：未采用需 Windows MSVC 编译的 InsightFace C++ 扩展；在相同硬件下用 Facenet 路线更易部署，适合毕设原型与实验说明。

### 5. `age_estimator.py`：年龄段估计

- **模型**：EfficientNet-B0，使用 ImageNet 预训练权重后在年龄数据集上微调。
- **类别**：采用更稳定的阶段型 5 类：儿童（0-12）、青少年（13-19）、青年（20-35）、中年（36-55）、老年（56+）。
- **训练数据**：`train_age_model.py` 支持 UTKFace 文件名年龄标签，也支持 FairFace CSV 年龄段标签。
- **运行行为**：若 `age_model_effnet_b0.pth` 不存在，Web 系统仍可正常识别人脸，但年龄段显示为“未加载”。
- **验证方式**：录入面容时可填写真实年龄；识别时系统会比较预测年龄段与真实年龄所在年龄段，并显示“预测正确/预测错误”。
- **场景校准**：考虑训练集人群与实际黄种人场景差异，系统会将模型输出年龄段自动上调一档。
- **稳定输出**：年龄段结果使用最近 `AGE_SMOOTHING_WINDOW` 帧多数投票，减少单帧预测抖动。

### 6. 档案文件 `face_profiles_v2.npz`

- 保存字段：`version`（当前为 2）、`person1` 至 `person10`（各为 512 维 `float32` 向量，未录入则不保存）。
- 旧版 HOG 等非 v2 格式不会自动载入，需**重新录入**。

---

## 运行方式

### 人体检测 + 深度（OpenCV）

```bash
cd FYP
conda activate realsense   # 或你的环境
python test.py
```

- 图像窗口 **获得焦点** 后按 `q` / `Q` / `Esc` 退出；也可关闭窗口退出。

### Web 人脸录入与识别

```bash
python face_app.py
```

浏览器访问：**http://127.0.0.1:5000**

### PhysFormer 训练命令（UBFC 子集 + VIPL 预训练微调）

```powershell
conda activate realsense   # 或已安装 torch 的独立环境
cd physformer\PhysFormer
python train_Physformer_160_UBFC.py ^
  --ubfc_root "..\..\datasets\UBFC-rPPG" ^
  --pretrained "..\..\weights\Physformer_VIPL_fold1.pkl" ^
  --gpu 0 --epochs 25 --log Physformer_UBFC_finetune
```

说明：`--ubfc_root` 指向包含若干 `subject*` 子文件夹的根目录；`--clip_stride` 可按显存与磁盘自行调整（见 `train_Physformer_160_UBFC.py` 参数）。

- 至少完成 1 个面容录入后即可进入 **识别**；如需重新采集，可在对应人物卡片点击重新录入。
- 录入前请在对应人物卡片填写真实年龄；该年龄仅保存在本地档案中，用于验证年龄段预测是否正确。
- 终端 **Ctrl+C** 结束服务。

### 训练年龄段模型

使用 UTKFace：

```bash
python train_age_model.py --device cuda --epochs 16 --warmup-epochs 2 --batch-size 32 --num-workers 4
```

当前项目已将默认训练目录配置为 `UTKFace`。如需指定其他目录，可使用 `--utkface-root` 覆盖。

使用 FairFace：

```bash
python train_age_model.py --device cuda --fairface-root path/to/fairface --fairface-csv path/to/fairface_label_train.csv --epochs 16 --warmup-epochs 2 --batch-size 32 --num-workers 4
```

训练完成后会生成 `age_model_effnet_b0.pth`，再次启动 `face_app.py` 即会自动加载。
训练脚本会先冻结主干训练分类头，再解冻全模型微调，并输出普通准确率、相邻年龄段容错准确率和宏平均召回率。

---

## 配置说明（`config.py`）

| 类别 | 关键参数 | 含义 |
|------|-----------|------|
| 相机 | `WIDTH`, `HEIGHT`, `FPS` | 流分辨率与帧率 |
| YOLO | `YOLO_MODEL`, `YOLO_CONFIDENCE`, `YOLO_IMGSZ`, `YOLO_DEVICE` | 模型与推理设置 |
| 追踪 | `TRACK_IOU_THRESHOLD`, `TRACK_MAX_MISSES`, `TRACK_DRAW_MAX_MISSES` | 简单追踪稳定性 |
| 深度 | `DEPTH_VALID_MIN_M`, `DEPTH_VALID_MAX_M`, `DEPTH_MIN_VALID_PIXELS` | ROI 内参与中值统计的像素条件 |
| 深度 | `DEPTH_VITAL_OPTIMAL_MIN_M`, `DEPTH_VITAL_OPTIMAL_MAX_M` | 推荐站位区间（米），用于 `depth_zone` 与界面提示 |
| 深度 | `DEPTH_VIS_ALPHA` | 深度伪彩可视化缩放（仅显示，不影响测距） |
| 融合 | `FUSION_DEPTH_SCENE_MIN_M`, `FUSION_DEPTH_SCENE_MAX_M` | 深度映射到雷达通道时的场景近/远界（米）；可被环境变量覆盖 |
| 融合 | `FUSION_USE_DEPTH_BIN_MATCH` | 是否启用「深度→雷达距离门 id」贪心匹配（否则为名次配对） |
| 雷达上位机 | `RADAR_PI_BASE` | 树莓派 `shumeipai.py` 的根 URL，供轮询 `/api/radar` 与页面代理 |
| 人脸 | `FACE_RECOGNITION_THRESHOLD` | 识别相似度阈值（调高更严） |
| 人脸 | `FACE_MAX_PROFILES`, `FACE_DUPLICATE_THRESHOLD` | 最大面容档案数与重复面容判定阈值 |
| 人脸 | `FACE_MIN_MTCNN_PROB`, `FACE_MIN_BLUR_VARIANCE` | 检测与清晰度门控 |
| 录入 | `ENROLL_SAMPLES_TARGET`, `ENROLL_SAMPLE_INTERVAL_SEC` | 采样帧数与间隔 |
| 年龄 | `AGE_MODEL_FILE`, `AGE_INPUT_SIZE`, `AGE_CLASSES` | 年龄模型权重、输入尺寸与年龄段定义 |
| 年龄 | `AGE_CALIBRATION_SHIFT`, `AGE_SMOOTHING_WINDOW` | 年龄段上调校准与多帧多数投票窗口 |
| 年龄训练 | `AGE_TRAIN_UTKFACE_ROOT` | 默认 UTKFace 训练数据目录 |

---

## 伦理与数据

- 人脸数据仅保存在本地 **`face_profiles_v2.npz`**，不向第三方上传。
- 毕设论文中建议说明：数据用途、存储位置与删除方式。

---

## 常见问题

- **按 `q` 无反应**：请先点击 **OpenCV 图像窗口** 再按键，焦点在终端时无法捕获。
- **识别总为 Unknown**：可适当 **降低** `FACE_RECOGNITION_THRESHOLD`，或在光照稳定、正对镜头下重新录入。
- **YOLO 很慢**：可将 `YOLO_IMGSZ` 调小，或在使用 NVIDIA GPU 时将 `YOLO_DEVICE` 设为 `0`。

---

## 小结

本项目构成一套 **RGB-D 摄像头 +（可选）树莓派毫米波雷达** 的非接触式生命体征与人脸识别原型，适用于毕设演示与实验记录，**非医疗诊断用途**。

- **视觉侧（上位机）**：Intel RealSense D435 提供 **彩色与深度对齐流**；在人脸 ROI 上估计 **深度中值（米）**、**有效深度占比**，并给出相对推荐站位的 **距离分区**（`depth_zone`）。Web 端并列展示 **RGB 与深度伪彩**（`/video_feed` 与 `/video_depth`）。人脸分支采用 Facenet 路线完成 **最多 10 人建档、识别与重复面容检测**；可选 **PhysFormer** 输出 **每人 rPPG 心率**，并以绿通道 FFT 提供 **辅助呼吸率**。
- **雷达侧（树莓派）**：`shumeipai.py` 从串口读取相位剖面，经 **带通、谐波抑制、Hann 窗 FFT + 抛物线细化、自相关交叉校验与 EMA**，输出 **主通道** 及 **多距离门** 的呼吸/心率与质量指标；通过 HTTP **`/api/radar`** 提供给上位机，**不涉及修改本 README 未列出的串口协议时仅替换算法即可迭代**。
- **跨模态融合**：`radar_fusion.py` 将 **深度排序** 与 **雷达通道** 对齐：默认按场景深度区间把人物映射到 **首选距离门 id**，再以贪心策略分配通道；融合输出 **视觉 HR/RR、雷达 HR/RR、加权融合值**，并对两路分歧做 **置信度软化**。相关可调参数见 `config.py`（如 `FUSION_DEPTH_SCENE_*`、`FUSION_USE_DEPTH_BIN_MATCH`、`RADAR_PI_BASE`）。
- **论文撰写提示**：明确写出 **启发式前提**（深度↔雷达 bin 对应依赖场景与雷达安装）、**各模态局限**（光照、运动、多径）及 **与金标准设备的对比方法**（若有）。

---

## 深度相机参数与环境变量（答辩 / 复现）

本节汇总 **RealSense 深度参与融合与界面** 时的可调项，并补充 **树莓派雷达服务**、**fyp_launcher 一键入口** 的环境变量，便于双机部署与答辩复现。相机底层分辨率仍为 `WIDTH`×`HEIGHT`@`FPS`，深度与彩色 **对齐到彩色**（见上文「实现细节」）。

### `config.py` 中与深度直接相关的量

| 符号 | 含义 |
|------|------|
| `DEPTH_VALID_MIN_M` / `DEPTH_VALID_MAX_M` | 人脸 ROI 内参与 **中值深度** 统计的有效深度范围（米） |
| `DEPTH_MIN_VALID_PIXELS` | ROI 内至少多少个有效深度像素才认为测距可用 |
| `DEPTH_VITAL_OPTIMAL_MIN_M` / `DEPTH_VITAL_OPTIMAL_MAX_M` | **推荐站位**区间；人脸深度落在此区间外时 `depth_zone` 为 `too_near` / `too_far` |
| `DEPTH_VIS_ALPHA` | OpenCV 深度图转 8 位时的缩放系数，**仅影响伪彩显示**，不改变 `depth_m` 数值 |
| `FUSION_DEPTH_SCENE_MIN_M` / `FUSION_DEPTH_SCENE_MAX_M` | 将场景深度 **线性映射** 到雷达多通道下标 `0…N-1` 时的 **近端 / 远端**（米）；需结合实验室布局调节 |
| `FUSION_USE_DEPTH_BIN_MATCH` | `True`：按深度首选通道 id + **贪心分配**；`False`：退回「按深度排序名次 ↔ 雷达通道名次」 |

### 可通过环境变量覆盖的项（上位机）

在启动 `face_app.py` 前设置（Linux/macOS 用 `export`，Windows PowerShell 用 `$env:VAR="value"`）：

| 环境变量 | 作用 |
|----------|------|
| `RADAR_PI_BASE` | 例如 **`http://10.245.232.43:5000`**（与本仓库默认树莓派 IP 一致），为空则不拉取雷达 JSON |
| `FUSION_DEPTH_SCENE_MIN_M` / `FUSION_DEPTH_SCENE_MAX_M` | 覆盖融合用的场景深度窗（米） |
| `FUSION_USE_DEPTH_BIN_MATCH` | 设为 `0` 或 `false` 关闭深度–通道 id 匹配 |
| `PHYSFORMER_ENABLED` | `0` 关闭 PhysFormer 心率 |
| `PHYSFORMER_WEIGHTS` | 指定 `.pkl` 权重路径 |

### 树莓派雷达服务（`shumeipai.py`）环境变量

在 **树莓派** 上启动雷达 HTTP 服务前可设置（默认串口路径适用于 Linux；Windows 若本地跑调试版则需改为 `COMx` 等）：

| 环境变量 | 默认 / 说明 |
|----------|-------------|
| `FYP_PC_HOST` | 上位机局域网 IP；未单独设置 `FYP_UI_URL` / `FYP_LAUNCHER_URL` 时用于拼出界面与 launcher 地址 |
| `FYP_UI_URL` | 上位机 `face_app` 根地址（可与 `FYP_PC_HOST` 联动生成） |
| `FYP_LAUNCHER_URL` | 上位机 `fyp_launcher` 地址（可与 `FYP_PC_HOST` 联动生成），供 Pi **POST /launch** |
| `FYP_CAMERA_ENABLE` | 默认 `1`（摄像头插在树莓派时）；设为 `0` 仅雷达、不采集摄像头 |
| `FYP_CAMERA_MODE` | 默认 **`auto`**：优先 **RealSense**（`pyrealsense2`，与 `jianjie.py` 一致），失败再试 UVC；`realsense` / `uvc` 可强制单一路径 |
| `FYP_CAMERA_INDEX` / `FYP_CAMERA_WIDTH` / `FYP_CAMERA_HEIGHT` / `FYP_CAMERA_FPS` | UVC 时的设备索引与分辨率、帧率（RealSense 启动时也使用相同宽高与帧率） |
| `FYP_CAMERA_DEVICE` | 例如 **`/dev/video2`**：直接打开该 V4L2 节点（RealSense 在部分系统上会生成多个 video 节点） |
| `FYP_CAMERA_PROBE` | 默认 `1`：依次尝试打开能出画的 `/dev/video*`；设为 `0` 则只用 `FYP_CAMERA_INDEX` |
| `FYP_REALSENSE_COLOR_ONLY` | 默认 `0`；设为 `1` 时跳过深度流（减轻带宽，便于 USB2 或调试） |
| `FYP_LAUNCHER_TOKEN` | 与 launcher 校验头 `X-FYP-Token` 一致时，远程唤醒更安全 |
| `FYP_SERIAL` | 毫米波串口设备路径，默认 `/dev/ttyUSB0` |
| `FYP_BIND` | HTTP 监听地址，默认 `0.0.0.0` |
| `FYP_PORT` | HTTP 端口，默认 `5000`（与上位机 `RADAR_PI_BASE` 端口一致） |
| `FYP_OPEN_KIOSK` | 设为非 `1` 时可关闭启动时尝试打开 Chromium 全屏等行为 |
| `FYP_AUTO_START` | 默认 `1`：启动序列（POST launcher、可选 kiosk）；设为 `0` 可关闭 |

### 上位机一键入口（`fyp_launcher.py`）环境变量

用于 **`python fyp_launcher.py`** 同时拉起 `face_app` 并常驻轻量 HTTP，供树莓派回调：

| 环境变量 | 默认 / 说明 |
|----------|-------------|
| `RADAR_PI_BASE` | 传给子进程 `face_app`：树莓派雷达 API 根 URL；未设置时 launcher 可能注入 `FYP_DEFAULT_RADAR_PI_BASE` 或内置默认 IP |
| `FYP_DEFAULT_RADAR_PI_BASE` | 覆盖内置默认树莓派地址（仅当未设置 `RADAR_PI_BASE` 时写入子进程） |
| `FYP_LAUNCHER_PORT` | Launcher 监听端口，默认 `8787` |
| `FYP_LAUNCHER_TOKEN` | Pi 请求 launcher 受保护接口时的令牌，需与 Pi 侧 `FYP_LAUNCHER_TOKEN` 一致 |
| `FYP_FACEAPP_CHECK` | 探测 `face_app` 是否就绪的 URL，默认 `http://127.0.0.1:5000/api/status` |
| `FYP_SKIP_FACE_APP_BOOT` | 设为 `1` / `true` / `yes` 时**仅启动 launcher**，不自动拉起 `face_app`（调试） |
| `FYP_EXIT_KILL_FACE` | 默认会随 launcher 退出结束本进程启动的 `face_app`；设为 `0` 则保留 `face_app` |

### HTTP `/api/status` 中的 `depth_camera` 字段

便于前端与答辩演示时核对当前策略，无需翻代码：

- `optimal_range_m`：对应 `DEPTH_VITAL_*` 的推荐站位；
- `fusion_scene_m`：当前用于映射雷达通道的场景深度窗；
- `fusion_use_depth_bin_match`：是否启用深度–雷达门匹配。

### 答辩时可强调的一点

**深度米数 ↔ 雷达 range bin** 无通用物理常数，依赖雷达安装高度、朝向与场景反射；上述 **场景深度窗** 宜在实测后写入论文「实验设置」，并说明采用 **启发式匹配** 而非唯一真值对应。

---

## 推送到 GitHub

### 1. 本机需已安装 Git

- Windows：安装 [Git for Windows](https://git-scm.com/download/win)，安装时勾选 **“Add Git to PATH”**。  
- 在 **PowerShell** 或 **Git Bash** 中执行下文命令（本仓库已含 `.git`，无需 `git init`）。

若 `git` 仍无法识别，可重启终端，或使用 **GitHub Desktop**（图形界面完成 clone / commit / push）。

### 2. 在 GitHub 上新建空仓库

1. 打开 [https://github.com/new](https://github.com/new)  
2. 填写 **Repository name**（例如 `FYP`）  
3. 可见性选 **Private**（含人脸/实验相关代码时建议私有）或 **Public**  
4. **不要**勾选 “Add a README / .gitignore / license”（本地已有文件），点击 **Create repository**。

### 3. 首次绑定远程并推送（HTTPS）

将下面命令里的 `你的用户名` 和 `仓库名` 换成你自己的（HTTPS 地址以 GitHub 新建页为准）：

```powershell
cd C:\Users\gsnlk\Desktop\FYP

git status
git add .
git commit -m "Initial commit: FYP RealSense + radar fusion project"

git remote add origin https://github.com/你的用户名/仓库名.git
git branch -M main
git push -u origin main
```

推送时若提示登录：

- **HTTPS**：用户名填 GitHub 用户名，密码处填 **Personal Access Token**（[Settings → Developer settings → Tokens](https://github.com/settings/tokens) 新建，勾选 `repo`）。  
- 或使用 **SSH**：先在 GitHub 添加 SSH 公钥，再把 `origin` 改为 `git@github.com:你的用户名/仓库名.git`。

### 4. 已加强的忽略规则（勿把隐私与大文件传上去）

`.gitignore` 已包含：`datasets/`、大权重 `*.pkl`、**人脸档案** `face_profiles*.npz`、**年龄模型** `age_model_effnet_b0.pth`、**YOLO 权重** `yolov8n.pt`、`.env`、以及 **`*.jpg` / `*.jpeg`**（减小克隆体积）等。  
若你曾把上述文件 **已经 commit 过**，需先从索引或历史中移除再推送。

### 5. 不再跟踪已上传的 JPG/JPEG（减轻今后 clone 体积）

仓库根目录已忽略 `*.jpg`、`*.jpeg`。若这些文件 **曾经被提交到 Git**，仅改 `.gitignore` 不会自动去掉远端里的旧版本，需要 **先从索引取消跟踪**（本地文件保留）：

```powershell
cd C:\Users\gsnlk\Desktop\FYP
powershell -ExecutionPolicy Bypass -File .\scripts\untrack_jpg_from_git.ps1
git add .gitignore
git commit -m "chore: stop tracking jpg/jpeg images"
git push
```

**说明**：上述步骤只会让 **之后的新克隆** 不再包含这些图片文件（它们会从仓库最新快照中删除）；别人仍可能在 **历史旧 commit** 里看到大文件。若必须把整个历史中 JPG 也抹掉以减小体积，需在备份仓库后使用 [`git filter-repo`](https://github.com/newren/git-filter-repo) 等重写历史（小组协作仓库慎用）。

### 6. 可选：GitHub CLI

若已安装 [`gh`](https://cli.github.com/)：

```powershell
cd C:\Users\gsnlk\Desktop\FYP
gh auth login
gh repo create FYP --private --source=. --remote=origin --push
```

---

## 许可与引用

- **Ultralytics YOLO**、**facenet-pytorch**、**Intel RealSense** 各自遵循其开源协议；撰写论文时请按课程要求引用对应库与论文。
