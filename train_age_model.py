import argparse
import csv
import random
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset, WeightedRandomSampler
from torchvision import transforms

from age_estimator import create_efficientnet_b0
from config import (
    AGE_CLASSES,
    AGE_INPUT_SIZE,
    AGE_MODEL_FILE,
    AGE_TRAIN_FAIRFACE_CSV,
    AGE_TRAIN_FAIRFACE_ROOT,
    AGE_TRAIN_UTKFACE_ROOT,
)


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def age_to_class_index(age: int) -> int:
    if age <= 12:
        return 0
    if age <= 19:
        return 1
    if age <= 35:
        return 2
    if age <= 55:
        return 3
    return 4


def fairface_age_to_midpoint(age_text: str) -> Optional[int]:
    text = age_text.strip().lower()
    if not text:
        return None
    if "more" in text or "70" in text and "+" in text:
        return 75
    if "-" in text:
        left, right = text.replace(" ", "").split("-", 1)
        if left.isdigit() and right.isdigit():
            return (int(left) + int(right)) // 2
    if text.isdigit():
        return int(text)
    return None


class UTKFaceDataset(Dataset):
    def __init__(self, root: str, transform=None):
        self.root = Path(root)
        self.transform = transform
        self.samples = []

        for path in self.root.rglob("*"):
            if path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            age_part = path.name.split("_", 1)[0]
            if not age_part.isdigit():
                continue
            age = int(age_part)
            self.samples.append((path, age_to_class_index(age)))

        if not self.samples:
            raise ValueError(f"UTKFace 数据目录中没有找到有效图片: {self.root}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label


class FairFaceDataset(Dataset):
    def __init__(self, root: str, csv_path: str, transform=None):
        self.root = Path(root)
        self.csv_path = Path(csv_path)
        self.transform = transform
        self.samples = []

        with self.csv_path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                file_name = row.get("file") or row.get("path") or row.get("image")
                age_text = row.get("age") or row.get("age_group") or row.get("Age")
                if not file_name or not age_text:
                    continue
                age = fairface_age_to_midpoint(age_text)
                if age is None:
                    continue
                path = self.root / file_name
                if path.exists():
                    self.samples.append((path, age_to_class_index(age)))

        if not self.samples:
            raise ValueError(f"FairFace CSV 中没有找到有效样本: {self.csv_path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label


def split_dataset(dataset, val_ratio: float, seed: int) -> Tuple[Subset, Subset]:
    indices = list(range(len(dataset)))
    random.Random(seed).shuffle(indices)
    val_size = max(1, int(len(indices) * val_ratio))
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]
    return Subset(dataset, train_indices), Subset(dataset, val_indices)


def labels_from_subset(subset: Subset) -> List[int]:
    labels = []
    for index in subset.indices:
        labels.append(subset.dataset.samples[index][1])
    return labels


def make_weighted_sampler(labels: List[int]):
    counts = [max(1, labels.count(i)) for i in range(len(AGE_CLASSES))]
    weights = [1.0 / counts[label] for label in labels]
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


def make_class_weights(labels: List[int], device):
    counts = torch.tensor(
        [max(1, labels.count(i)) for i in range(len(AGE_CLASSES))],
        dtype=torch.float32,
    )
    weights = counts.sum() / (counts * len(AGE_CLASSES))
    return weights.to(device)


def set_backbone_trainable(model, trainable: bool):
    for param in model.features.parameters():
        param.requires_grad = trainable


def evaluate(model, loader, device):
    model.eval()
    correct = 0
    adjacent_correct = 0
    total = 0
    class_correct = [0 for _ in AGE_CLASSES]
    class_total = [0 for _ in AGE_CLASSES]
    confusion = [[0 for _ in AGE_CLASSES] for _ in AGE_CLASSES]

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(images)
            preds = torch.argmax(logits, dim=1)
            correct += int((preds == labels).sum().item())
            adjacent_correct += int((torch.abs(preds - labels) <= 1).sum().item())
            total += int(labels.numel())
            for label, pred in zip(labels.cpu().tolist(), preds.cpu().tolist()):
                class_total[label] += 1
                confusion[label][pred] += 1
                if label == pred:
                    class_correct[label] += 1

    accuracy = correct / total if total else 0.0
    adjacent_accuracy = adjacent_correct / total if total else 0.0
    recalls = [
        class_correct[i] / class_total[i] if class_total[i] else 0.0
        for i in range(len(AGE_CLASSES))
    ]
    macro_recall = sum(recalls) / len(recalls)
    return accuracy, adjacent_accuracy, macro_recall, confusion


def build_datasets(args, train_transform, val_transform):
    train_datasets = []
    val_datasets = []

    if args.utkface_root:
        full_train = UTKFaceDataset(args.utkface_root, transform=train_transform)
        full_val = UTKFaceDataset(args.utkface_root, transform=val_transform)
        train_subset, val_subset = split_dataset(full_train, args.val_ratio, args.seed)
        _, val_indices = split_dataset(full_val, args.val_ratio, args.seed)
        train_datasets.append(train_subset)
        val_datasets.append(val_indices)

    if args.fairface_root and args.fairface_csv:
        full_train = FairFaceDataset(args.fairface_root, args.fairface_csv, transform=train_transform)
        full_val = FairFaceDataset(args.fairface_root, args.fairface_csv, transform=val_transform)
        train_subset, val_subset = split_dataset(full_train, args.val_ratio, args.seed)
        _, val_indices = split_dataset(full_val, args.val_ratio, args.seed)
        train_datasets.append(train_subset)
        val_datasets.append(val_indices)

    if not train_datasets:
        raise ValueError("请至少提供 --utkface-root 或 --fairface-root + --fairface-csv")

    train_dataset = train_datasets[0] if len(train_datasets) == 1 else ConcatDataset(train_datasets)
    val_dataset = val_datasets[0] if len(val_datasets) == 1 else ConcatDataset(val_datasets)
    return train_dataset, val_dataset


def get_subset_labels(dataset) -> List[int]:
    if isinstance(dataset, Subset):
        return labels_from_subset(dataset)
    labels = []
    for child in dataset.datasets:
        labels.extend(labels_from_subset(child))
    return labels


def main():
    parser = argparse.ArgumentParser(description="Train EfficientNet-B0 age-group classifier.")
    parser.add_argument("--utkface-root", default=AGE_TRAIN_UTKFACE_ROOT, help="UTKFace 图片目录")
    parser.add_argument("--fairface-root", default=AGE_TRAIN_FAIRFACE_ROOT, help="FairFace 图片根目录")
    parser.add_argument("--fairface-csv", default=AGE_TRAIN_FAIRFACE_CSV, help="FairFace CSV 标签文件")
    parser.add_argument("--output", default=AGE_MODEL_FILE, help="输出权重文件")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--warmup-epochs", type=int, default=2, help="先冻结主干训练分类头的 epoch 数")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--head-lr", type=float, default=8e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.04)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"], help="训练设备")
    parser.add_argument("--log-interval", type=int, default=50, help="每隔多少个 batch 打印一次进度")
    parser.add_argument("--no-pretrained", action="store_true", help="不使用 ImageNet 预训练")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("你指定了 --device cuda，但当前 PyTorch 检测不到 CUDA。")
    use_cuda = torch.cuda.is_available() if args.device == "auto" else args.device == "cuda"
    device = torch.device("cuda" if use_cuda else "cpu")
    if use_cuda:
        torch.backends.cudnn.benchmark = True
        print(f"Training device: CUDA - {torch.cuda.get_device_name(0)}")
        print(f"CUDA version from PyTorch: {torch.version.cuda}")
    else:
        print("Training device: CPU")

    train_transform = transforms.Compose(
        [
            transforms.Resize((AGE_INPUT_SIZE + 32, AGE_INPUT_SIZE + 32)),
            transforms.RandomResizedCrop(AGE_INPUT_SIZE, scale=(0.82, 1.0), ratio=(0.9, 1.1)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.22, contrast=0.22, saturation=0.16),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.12, scale=(0.02, 0.08), ratio=(0.3, 3.3)),
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.Resize((AGE_INPUT_SIZE, AGE_INPUT_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    train_dataset, val_dataset = build_datasets(args, train_transform, val_transform)
    train_labels = get_subset_labels(train_dataset)
    sampler = make_weighted_sampler(train_labels)
    print(f"Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)}")
    print("Class counts:", {AGE_CLASSES[i]["range"]: train_labels.count(i) for i in range(len(AGE_CLASSES))})

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=use_cuda,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=use_cuda,
        persistent_workers=args.num_workers > 0,
    )

    model = create_efficientnet_b0(len(AGE_CLASSES), pretrained=not args.no_pretrained).to(device)
    class_weights = make_class_weights(train_labels, device)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights, label_smoothing=args.label_smoothing)

    best_macro_recall = -1.0
    for epoch in range(1, args.epochs + 1):
        warmup = epoch <= args.warmup_epochs
        set_backbone_trainable(model, trainable=not warmup)
        lr = args.head_lr if warmup else args.lr
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr,
            weight_decay=args.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, len(train_loader)),
            eta_min=lr * 0.08,
        )
        model.train()
        running_loss = 0.0
        for batch_idx, (images, labels) in enumerate(train_loader, start=1):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            running_loss += float(loss.item()) * int(labels.numel())
            if args.log_interval > 0 and batch_idx % args.log_interval == 0:
                print(
                    f"Epoch {epoch:02d}/{args.epochs} "
                    f"batch {batch_idx}/{len(train_loader)} "
                    f"loss={float(loss.item()):.4f}"
                )

        train_loss = running_loss / max(1, len(train_dataset))
        val_acc, val_adj_acc, val_macro_recall, confusion = evaluate(model, val_loader, device)
        print(
            f"Epoch {epoch:02d}/{args.epochs} "
            f"phase={'head' if warmup else 'finetune'} "
            f"loss={train_loss:.4f} val_acc={val_acc:.4f} "
            f"adjacent_acc={val_adj_acc:.4f} macro_recall={val_macro_recall:.4f}"
        )

        if val_macro_recall > best_macro_recall:
            best_macro_recall = val_macro_recall
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "classes": AGE_CLASSES,
                    "input_size": AGE_INPUT_SIZE,
                    "best_macro_recall": best_macro_recall,
                    "val_accuracy": val_acc,
                    "val_adjacent_accuracy": val_adj_acc,
                    "confusion_matrix": confusion,
                },
                args.output,
            )
            print(f"Saved best checkpoint to {args.output}")


if __name__ == "__main__":
    main()
