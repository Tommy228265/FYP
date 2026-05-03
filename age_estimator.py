from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch
from PIL import Image

try:
    from torchvision import models, transforms
except ImportError:
    models = None
    transforms = None


def create_efficientnet_b0(num_classes: int, pretrained: bool = False):
    if models is None:
        raise ImportError("请安装 torchvision: pip install torchvision")

    try:
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)
    except AttributeError:
        model = models.efficientnet_b0(pretrained=pretrained)

    in_features = model.classifier[-1].in_features
    model.classifier[-1] = torch.nn.Linear(in_features, num_classes)
    return model


class AgeEstimator:
    """EfficientNet-B0 年龄段分类推理器。"""

    def __init__(
        self,
        model_path: str,
        classes: List[Dict[str, str]],
        input_size: int = 224,
        device: Optional[str] = None,
    ):
        self.model_path = Path(model_path)
        self.classes = classes
        self.input_size = input_size
        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.available = False
        self.status = "年龄模型未加载"
        self.model = None

        if transforms is not None:
            self.transform = transforms.Compose(
                [
                    transforms.Resize((input_size, input_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )
        else:
            self.transform = None

        self._load()

    def _load(self):
        if not self.model_path.exists():
            self.status = f"未找到年龄模型权重: {self.model_path}"
            return

        try:
            checkpoint = torch.load(str(self.model_path), map_location=self.device)
            checkpoint_classes = checkpoint.get("classes") if isinstance(checkpoint, dict) else None
            if checkpoint_classes:
                self.classes = checkpoint_classes
            class_count = len(self.classes)
            model = create_efficientnet_b0(class_count, pretrained=False)
            state_dict = checkpoint.get("model_state", checkpoint) if isinstance(checkpoint, dict) else checkpoint
            model.load_state_dict(state_dict)
            model.eval().to(self.device)
            self.model = model
            self.available = True
            self.status = f"年龄模型已加载（{class_count} 类）"
        except Exception as exc:
            self.status = f"年龄模型加载失败: {exc}"
            self.available = False

    @torch.no_grad()
    def estimate_from_bgr_crop(self, face_bgr: np.ndarray):
        if not self.available or self.model is None or self.transform is None:
            return {
                "label": "未加载",
                "label_en": "Age model not loaded",
                "range": "-",
                "class_index": -1,
                "class_id": "unavailable",
                "confidence": 0.0,
            }

        if face_bgr is None or face_bgr.size == 0:
            return {
                "label": "无法估计",
                "label_en": "N/A",
                "range": "-",
                "class_index": -1,
                "class_id": "invalid_crop",
                "confidence": 0.0,
            }

        rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        batch = self.transform(image).unsqueeze(0).to(self.device)
        logits = self.model(batch)
        probs = torch.softmax(logits, dim=1).cpu().numpy().reshape(-1)
        index = int(np.argmax(probs))
        confidence = float(probs[index])
        item = self.classes[index] if index < len(self.classes) else {}
        return {
            "label": item.get("label", f"年龄段{index}"),
            "label_en": item.get("label_en", f"Age {index}"),
            "range": item.get("range", "-"),
            "class_index": index,
            "class_id": item.get("id", f"age_{index}"),
            "confidence": confidence,
        }
