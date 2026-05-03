"""
人脸嵌入：facenet-pytorch (MTCNN + InceptionResnetV1)，避免 Windows 上 InsightFace 需 MSVC 编译。
"""
from __future__ import annotations

import cv2
import numpy as np
import torch
from PIL import Image

try:
    from facenet_pytorch import MTCNN, InceptionResnetV1
except ImportError as e:
    raise ImportError(
        "请安装: pip install facenet-pytorch torch torchvision"
    ) from e


def laplacian_blur_score(bgr: np.ndarray) -> float:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


class FaceEmbedder:
    """检测 + 对齐 + 512 维嵌入（L2 归一化）。"""

    def __init__(
        self,
        device: str | None = None,
        min_face_size: int = 60,
        pretrained: str = "vggface2",
    ):
        self.device = torch.device(
            device
            if device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.mtcnn = MTCNN(
            image_size=160,
            margin=14,
            min_face_size=min_face_size,
            thresholds=[0.6, 0.7, 0.7],
            factor=0.709,
            post_process=True,
            device=self.device,
            keep_all=True,
        )
        self.resnet = InceptionResnetV1(pretrained=pretrained).eval().to(self.device)

    @torch.no_grad()
    def embed_all_from_bgr(self, frame_bgr: np.ndarray):
        """
        返回同一帧内全部人脸，按画面从左到右排序。
        每一项包含 (embedding, mtcnn_prob, blur_score, bbox)，bbox 为 (x, y, w, h)。
        """
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)

        boxes, det_probs = self.mtcnn.detect(pil)
        if boxes is None:
            return []

        faces, probs = self.mtcnn(pil, return_prob=True)
        if faces is not None and faces.dim() == 3:
            faces = faces.unsqueeze(0)

        h, w = frame_bgr.shape[:2]
        det_probs_arr = np.asarray(det_probs).flatten() if det_probs is not None else np.array([])
        probs_arr = np.asarray(probs).flatten() if probs is not None else np.array([])
        embeddings = []

        if faces is not None and faces.dim() == 4:
            face_batch = faces.to(self.device)
            emb_batch = self.resnet(face_batch).cpu().numpy().astype(np.float32)
            norms = np.linalg.norm(emb_batch, axis=1, keepdims=True) + 1e-8
            emb_batch = emb_batch / norms
        else:
            emb_batch = None

        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = [int(round(v)) for v in box]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            crop = frame_bgr[y1:y2, x1:x2]
            blur = laplacian_blur_score(crop) if crop.size else 0.0
            det_prob = float(det_probs_arr[i]) if i < det_probs_arr.size else None
            prob = float(probs_arr[i]) if i < probs_arr.size else det_prob
            emb = emb_batch[i] if emb_batch is not None and i < emb_batch.shape[0] else None
            embeddings.append((emb, prob, blur, (x1, y1, x2 - x1, y2 - y1)))

        return sorted(embeddings, key=lambda item: item[3][0])

    @torch.no_grad()
    def embed_from_bgr(self, frame_bgr: np.ndarray):
        """
        兼容旧调用：取面积最大的人脸。
        返回 (embedding, mtcnn_prob, blur_score, bbox)。
        """
        faces = self.embed_all_from_bgr(frame_bgr)
        if not faces:
            return None, None, 0.0, None
        return max(faces, key=lambda item: item[3][2] * item[3][3])
