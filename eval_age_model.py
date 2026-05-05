"""
Evaluate the age-group classifier (EfficientNet-B0) on the same val split as training.

For your thesis: report val accuracy, adjacent accuracy, macro recall, and optionally
export the confusion matrix. Use the same --val-ratio and --seed as `train_age_model.py`
so the split matches the checkpoint you saved.

Other models in this project (no standalone test here):
- YOLOv8: cite COCO person AP from Ultralytics or run `yolo val` on COCO.
- MTCNN + FaceNet: cite LFW / VGGFace2 benchmarks from facenet-pytorch or original papers.
- PhysFormer: use validation MAE/RMSE from your `train_Physformer_160_UBFC.py` run or cite the paper.
- Visual respiration / radar DSP: not supervised classifiers; use physical or field trials instead.

Usage:
  python eval_age_model.py --checkpoint weights/age_model.pt --utkface-root D:/data/UTKFace
  python eval_age_model.py --checkpoint-only   # print metrics stored in checkpoint only
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List

import torch
from torch.utils.data import DataLoader
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
from train_age_model import build_datasets, evaluate


def _load_checkpoint(path: Path) -> Dict[str, Any]:
    try:
        return torch.load(str(path), map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(str(path), map_location="cpu")


def _build_model_from_checkpoint(ckpt: Dict[str, Any], device: torch.device):
    classes = ckpt.get("classes")
    if classes and isinstance(classes, list):
        n_cls = len(classes)
        label_names = classes
    else:
        n_cls = len(AGE_CLASSES)
        label_names = AGE_CLASSES
    state = ckpt.get("model_state", ckpt)
    model = create_efficientnet_b0(n_cls, pretrained=False)
    model.load_state_dict(state)
    model.eval()
    return model.to(device), label_names


def _label_text(c: Any) -> str:
    if isinstance(c, dict):
        return str(c.get("label_en") or c.get("range") or c.get("label", ""))
    return str(c)


def _print_confusion(confusion: List[List[int]], class_labels: List[Any]) -> None:
    names = [_label_text(c) for c in class_labels]
    if len(names) != len(confusion):
        names = [f"class_{i}" for i in range(len(confusion))]
    # header
    w = max(4, max(len(n) for n in names) if names else 4)
    header = "true\\pred".ljust(w) + "".join(n.ljust(w) for n in names)
    print(header)
    for i, row in enumerate(confusion):
        label = names[i] if i < len(names) else str(i)
        print(label.ljust(w) + "".join(str(x).ljust(w) for x in row))


def _save_confusion_csv(confusion: List[List[int]], path: Path, class_labels: List[Any]) -> None:
    names = [_label_text(c) for c in class_labels]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["true \\ pred"] + names)
        for i, row in enumerate(confusion):
            label = names[i] if i < len(names) else str(i)
            w.writerow([label] + row)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate age-group model on validation split.")
    parser.add_argument("--checkpoint", default=str(AGE_MODEL_FILE), help="Path to .pt checkpoint")
    parser.add_argument("--utkface-root", default=AGE_TRAIN_UTKFACE_ROOT, help="UTKFace image root")
    parser.add_argument("--fairface-root", default=AGE_TRAIN_FAIRFACE_ROOT, help="FairFace image root")
    parser.add_argument("--fairface-csv", default=AGE_TRAIN_FAIRFACE_CSV, help="FairFace CSV")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Same as training")
    parser.add_argument("--seed", type=int, default=42, help="Same as training for comparable split")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument(
        "--save-dir",
        default="",
        help="If set, write metrics.json and confusion_matrix.csv here",
    )
    parser.add_argument(
        "--checkpoint-only",
        action="store_true",
        help="Only print metrics stored in the checkpoint (no forward pass)",
    )
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_file():
        print(f"Checkpoint not found: {ckpt_path}", file=sys.stderr)
        sys.exit(1)

    ckpt = _load_checkpoint(ckpt_path)
    if not isinstance(ckpt, dict):
        print("Checkpoint is raw state_dict; use full training save format for metadata.", file=sys.stderr)
        sys.exit(1)

    if args.checkpoint_only:
        print("--- Metrics stored in checkpoint (from training) ---")
        for k in ("val_accuracy", "val_adjacent_accuracy", "best_macro_recall", "input_size"):
            if k in ckpt:
                print(f"  {k}: {ckpt[k]}")
        if "confusion_matrix" in ckpt:
            print("  (confusion_matrix present; re-run without --checkpoint-only to recompute on disk)")
        sys.exit(0)

    if not args.utkface_root and (not args.fairface_root or not args.fairface_csv):
        print(
            "Need data: set UTKFace and/or FairFace paths in config.py or pass "
            "--utkface-root / --fairface-root + --fairface-csv",
            file=sys.stderr,
        )
        sys.exit(1)

    use_cuda = torch.cuda.is_available() if args.device == "auto" else args.device == "cuda"
    device = torch.device("cuda" if use_cuda else "cpu")

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    val_transform = transforms.Compose(
        [
            transforms.Resize((AGE_INPUT_SIZE, AGE_INPUT_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    # build_datasets expects full args namespace with same fields as training
    train_transform = val_transform  # unused; build_datasets only needs val for us — but build_datasets builds both
    class _A:
        pass

    a = _A()
    a.utkface_root = args.utkface_root or None
    a.fairface_root = args.fairface_root or None
    a.fairface_csv = args.fairface_csv or None
    a.val_ratio = args.val_ratio
    a.seed = args.seed

    if not a.utkface_root and not (a.fairface_root and a.fairface_csv):
        print("No dataset path configured.", file=sys.stderr)
        sys.exit(1)

    _, val_dataset = build_datasets(a, val_transform, val_transform)
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=use_cuda,
    )

    model, class_labels = _build_model_from_checkpoint(ckpt, device)
    acc, adj_acc, macro_recall, confusion = evaluate(model, val_loader, device)

    n_val = len(val_dataset)
    print("--- Age model evaluation (validation split) ---")
    print(f"  checkpoint:     {ckpt_path.resolve()}")
    print(f"  val samples:    {n_val}")
    print(f"  val_ratio/seed: {args.val_ratio} / {args.seed}  (match training for same split)")
    print(f"  accuracy:            {acc:.4f}")
    print(f"  adjacent accuracy:   {adj_acc:.4f}")
    print(f"  macro recall:        {macro_recall:.4f}")
    print("--- Confusion (rows=true, cols=pred) ---")
    labels_for_print = class_labels if len(class_labels) == len(confusion) else AGE_CLASSES
    _print_confusion(confusion, labels_for_print)

    if args.save_dir:
        out = Path(args.save_dir)
        out.mkdir(parents=True, exist_ok=True)
        payload = {
            "checkpoint": str(ckpt_path.resolve()),
            "val_samples": n_val,
            "val_ratio": args.val_ratio,
            "seed": args.seed,
            "accuracy": acc,
            "adjacent_accuracy": adj_acc,
            "macro_recall": macro_recall,
            "age_classes": [c if isinstance(c, dict) else str(c) for c in (class_labels or AGE_CLASSES)],
        }
        (out / "age_model_val_metrics.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        _save_confusion_csv(confusion, out / "age_confusion_matrix.csv", labels_for_print)
        print(f"Wrote {out / 'age_model_val_metrics.json'} and confusion CSV.")

    train_val = ckpt.get("val_accuracy")
    if train_val is not None:
        print(f"(checkpoint recorded val_accuracy at save: {train_val:.4f})")


if __name__ == "__main__":
    main()
