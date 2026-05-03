# PhysFormer 训练：下载清单与数据目录（VIPL-HR + UBFC-rPPG）

本目录已包含从官方仓库获取的代码：`PhysFormer/`（与 [ZitongYu/PhysFormer](https://github.com/ZitongYu/PhysFormer) 一致）。

---

## 一、你必须下载的内容

### 1. VIPL-HR（训练与验证）

| 项目 | 说明 |
|------|------|
| **官方页面（中文）** | [VIPL-HR Database — CAS VIPL](https://vipl.ict.ac.cn/resources/databases/201811/t20181129_32716.html) |
| **英文索引** | [VIPL Databases](https://vipl.ict.ac.cn/en/resources/databases/) |
| **获取方式** | 下载页面上的 **Release Agreement**，由 **本校教职工（全职）** 签字并扫描，用 **学校/科研院所官方邮箱**（不接受 Gmail、163 等个人邮箱）发送至 **hanhu@ict.ac.cn**，通过后邮件获取下载链接。 |
| **引用** | 按 VIPL 页面要求引用 RhythmNet (TIP 2020) 与 VIPL-HR (ACCV 2018) 论文。 |

**重要：** PhysFormer 训练脚本读的不是原始 `.avi`，而是 **已裁剪的对齐人脸帧序列**（见下文「数据目录结构」）。官方数据包内通常带有说明与预处理脚本（常见为 Matlab）；需将视频转为与 `VIPL_fold1_train.txt` 中路径一致的 **PNG 序列**。

### 2. UBFC-rPPG（仅用于跨库测试，不参与本脚本默认训练）

| 项目 | 说明 |
|------|------|
| **官方主页** | [UBFC-rPPG — Y. Benezeth](https://sites.google.com/view/ybenezeth/ubfcrppg) |
| **内容** | 页面提供完整视频数据及读取 **ground truth（血氧仪）** 的 Matlab/Python 示例。 |
| **引用** | 使用数据集时按作者主页/论文要求标注引用。 |

下载完成后，跨库测试需要单独编写 DataLoader：将 UBFC 视频切成 **160 帧 × 128×128** 的片段（与 `Loadtemporal_data.py` 中 `clip_frames = 160` 一致），再加载你在 VIPL 上训练得到的 `Physformer_*.pkl` 做推理与 HR 误差统计。后续可在 `FYP` 仓库中增加 `eval_ubfc_physformer.py` 实现这一步。

### 3. PhysFormer 官方预训练与示例（可选，用于对照）

README 中提供的 **Google Drive**（测试样本与 fold1 权重）：

- 测试数据：[Google Drive](https://drive.google.com/file/d/1n1TpMQfU-OkZdJglEJyFp-vGo9JXbgsT/view?usp=sharing)
- 权重 `Physformer_VIPL_fold1.pkl`：[Google Drive](https://drive.google.com/file/d/1jBSbM88fA-beaoVi8ILFyL0SvVVMA9c9/view?usp=sharing)

可先跑通 `inference_OneSample_VIPL_PhysFormer.py`，确认环境与 GPU 正常，再跑完整 VIPL 训练。

---

## 二、VIPL 数据在磁盘上的目录结构（训练脚本要求）

训练脚本 `train_Physformer_160_VIPL.py` 使用：

```text
args.input_data + '/VIPL_frames/'
```

例如将 `--input_data` 设为 `D:\datasets\PhysFormer_VIPL`，则人脸帧应位于：

```text
D:\datasets\PhysFormer_VIPL\VIPL_frames\p1\v1\source1\image_00061.png
D:\datasets\PhysFormer_VIPL\VIPL_frames\p1\v1\source1\image_00062.png
...
```

`VIPL_fold1_train.txt` 每行第一列为相对路径（如 `p1/v1/source1`），第二列为起始帧号；代码会读取 `image_%05d.png`（见 `Loadtemporal_data.py` 中 `get_single_video_x`）。

因此：**原始 VIPL 下载后，必须预处理成上述 PNG 序列**，且路径层次与 fold 列表一致（列表文件已在 `PhysFormer/VIPL_fold1_train.txt`）。

---

## 三、Python 环境（建议单独 Conda 环境）

PhysFormer README 示例使用 **PyTorch 1.9** 与 **`pip install imgaug`**。你的机器若已安装较新 PyTorch，可先尝试直接运行；若报 API 不兼容，再新建环境例如：

```text
conda create -n physformer python=3.9 -y
conda activate physformer
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia
pip install imgaug opencv-python pandas scipy matplotlib
```

在 `PhysFormer` 目录下执行训练（**请先设置 `VIPL_ROOT` 为你的 `VIPL_frames` 的上一级路径**）：

```powershell
cd c:\Users\gsnlk\Desktop\FYP\physformer\PhysFormer
python train_Physformer_160_VIPL.py --input_data "D:\datasets\PhysFormer_VIPL" --gpu 0 --epochs 25 --log Physformer_VIPL_fold1_run1
```

`--input_data` 指向的目录下必须有子目录 **`VIPL_frames`**。

---

## 四、你本地当前进度

- [x] 已放置 PhysFormer 源码：`physformer/PhysFormer/`
- [ ] 向 VIPL 申请并下载数据，并完成 **VIPL_frames** 预处理
- [ ] 配置 Conda 环境并安装依赖
- [ ] 运行 `train_Physformer_160_VIPL.py` 得到 checkpoint
- [ ] 下载 UBFC-rPPG，并实现跨库推理脚本（下一步可写在 `FYP` 根目录）

数据到位并预处理完成后，即可在同一台机器上执行训练命令；若你希望，可在 VIPL 解压路径确定后把路径发给我，我可以帮你写 **Windows 下一键训练用的 `.ps1`**（仅替换 `--input_data`）。

---

## 五、仅用 UBFC-rPPG 训练 PhysFormer（无需 VIPL 签字）

若无法申请 VIPL-HR，可直接使用 **UBFC-rPPG**（见上文第三节下载）。目录示例（与 [rPPG-Toolbox](https://github.com/ubicomplab/rPPG-Toolbox) 一致）：

```text
UBFC_ROOT/
  subject1/vid.avi
  subject1/ground_truth.txt
  subject2/...
```

在项目内已提供脚本：

- `PhysFormer/Loadtemporal_data_UBFC.py`：从视频 + `ground_truth.txt` 生成 160 帧人脸块（OpenCV Haar 人脸检测）与标签；
- `PhysFormer/train_Physformer_160_UBFC.py`：训练入口；
- `run_train_ubfc.ps1`：PowerShell 快捷方式。

**命令示例（在 `physformer/PhysFormer` 下）：**

```powershell
conda activate <你的pytorch环境>
cd c:\Users\gsnlk\Desktop\FYP\physformer\PhysFormer
python train_Physformer_160_UBFC.py --ubfc_root "D:\datasets\UBFC-rPPG" --gpu 0 --epochs 25 --log Physformer_UBFC_run1
```

可选：使用官方 VIPL fold1 预训练权重做微调：

```powershell
python train_Physformer_160_UBFC.py --ubfc_root "D:\datasets\UBFC-rPPG" --pretrained ".\Physformer_VIPL_fold1.pkl" --log Physformer_UBFC_finetune
```

（请先将 Google Drive 上的 `Physformer_VIPL_fold1.pkl` 放到当前目录或写明路径。）

**说明：** UBFC 与视频的同步并非完美；若 loss 异常，可适当增大 `--clip_stride` 或减少难度（论文中可如实说明数据集局限）。
