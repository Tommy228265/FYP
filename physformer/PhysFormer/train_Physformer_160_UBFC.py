"""
Train PhysFormer on UBFC-rPPG (face crops from vid.avi + ground_truth.txt).
Place dataset under e.g. D:/data/UBFC-rPPG/subject1/vid.avi, ground_truth.txt
"""
from __future__ import print_function, division

import argparse
import math
import os

import scipy.io as sio
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

from Loadtemporal_data import Normaliztion, RandomHorizontalFlip, ToTensor
from Loadtemporal_data_UBFC import (
    build_manifest_for_subjects,
    discover_ubfc_subjects,
    split_subject_dirs,
    UBFC_train,
)
from model import ViT_ST_ST_Compact3_TDC_gra_sharp
from TorchLossComputer import TorchLossComputer


class Neg_Pearson(nn.Module):
    def __init__(self):
        super(Neg_Pearson, self).__init__()

    def forward(self, preds, labels):
        loss = 0
        for i in range(preds.shape[0]):
            sum_x = torch.sum(preds[i])
            sum_y = torch.sum(labels[i])
            sum_xy = torch.sum(preds[i] * labels[i])
            sum_x2 = torch.sum(torch.pow(preds[i], 2))
            sum_y2 = torch.sum(torch.pow(labels[i], 2))
            n = preds.shape[1]
            pearson = (n * sum_xy - sum_x * sum_y) / (
                torch.sqrt(
                    (n * sum_x2 - torch.pow(sum_x, 2)) * (n * sum_y2 - torch.pow(sum_y, 2))
                )
                + 1e-8
            )
            loss += 1 - pearson
        return loss / preds.shape[0]


class AvgrageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


def _ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def train_ubfc():
    global args
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this training script (PhysFormer on GPU).")

    _ensure_dir(args.log)
    log_file = open(os.path.join(args.log, args.log + "_log.txt"), "w", encoding="utf-8")

    _dev = torch.cuda.current_device()
    _name = torch.cuda.get_device_name(_dev)
    try:
        _free, _total = torch.cuda.mem_get_info()
        _mem = " | VRAM free %.1f / total %.1f GiB" % (
            _free / (1024.0**3),
            _total / (1024.0**3),
        )
    except Exception:
        _mem = ""
    print("Using GPU cuda:%d — %s%s" % (_dev, _name, _mem))
    log_file.write("device=cuda:%d %s\n" % (_dev, _name))
    log_file.flush()

    subject_dirs = discover_ubfc_subjects(args.ubfc_root)
    if not subject_dirs:
        raise FileNotFoundError(
            "No ground_truth.txt found under --ubfc_root: %s" % args.ubfc_root
        )

    train_subj, val_subj = split_subject_dirs(
        subject_dirs, val_ratio=args.val_ratio, seed=args.split_seed
    )
    if not train_subj:
        raise RuntimeError("No training subjects after split. Add more UBFC videos or lower --val_ratio.")
    train_manifest = build_manifest_for_subjects(
        train_subj,
        clip_stride=args.clip_stride,
        shuffle_starts=True,
        rng_seed=args.split_seed,
    )
    val_manifest = build_manifest_for_subjects(
        val_subj,
        clip_stride=max(args.clip_stride, 160),
        shuffle_starts=False,
        rng_seed=args.split_seed + 1,
    )

    n_flat_train = sum(len(m["starts"]) for m in train_manifest)
    n_flat_val = sum(len(m["starts"]) for m in val_manifest)
    print(
        "UBFC subjects: %d | train clips: %d | val subjects: %d | val clips: %d"
        % (len(subject_dirs), n_flat_train, len(val_subj), n_flat_val)
    )
    log_file.write(
        "subjects=%d train_clips=%d val_subj=%d val_clips=%d\n"
        % (len(subject_dirs), n_flat_train, len(val_subj), n_flat_val)
    )
    log_file.flush()

    model = ViT_ST_ST_Compact3_TDC_gra_sharp(
        image_size=(160, 128, 128),
        patches=(4, 4, 4),
        dim=96,
        ff_dim=144,
        num_heads=4,
        num_layers=12,
        dropout_rate=0.1,
        theta=0.7,
    )
    if args.pretrained:
        state = torch.load(args.pretrained, map_location="cuda")
        model.load_state_dict(state, strict=True)
        print("Loaded pretrained:", args.pretrained)
        log_file.write("pretrained=%s\n" % args.pretrained)
        log_file.flush()

    model = model.cuda()
    lr = args.lr
    optimizer1 = optim.Adam(model.parameters(), lr=lr, weight_decay=0.00005)
    scheduler1 = optim.lr_scheduler.StepLR(
        optimizer1, step_size=args.step_size, gamma=args.gamma
    )

    criterion_Pearson = Neg_Pearson()

    echo_batches = args.echo_batches
    a_start = 0.1
    b_start = 1.0
    exp_a = 0.5
    exp_b = 5.0
    fold_index = 1

    train_tf = transforms.Compose(
        [Normaliztion(), RandomHorizontalFlip(), ToTensor()]
    )
    val_tf = transforms.Compose([Normaliztion(), ToTensor()])

    for epoch in range(args.epochs):
        scheduler1.step()
        if (epoch + 1) % args.step_size == 0:
            lr *= args.gamma

        loss_rPPG_avg = AvgrageMeter()
        loss_peak_avg = AvgrageMeter()
        loss_kl_avg_test = AvgrageMeter()
        loss_hr_mae = AvgrageMeter()

        model.train()
        train_dl = UBFC_train(train_manifest, transform=train_tf)
        dataloader_train = DataLoader(
            train_dl,
            batch_size=args.batchsize,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
        )

        for i, sample_batched in enumerate(dataloader_train):
            inputs = sample_batched["video_x"].cuda()
            ecg = sample_batched["ecg"].cuda()
            clip_average_HR = sample_batched["clip_average_HR"].cuda()
            frame_rate = sample_batched["frame_rate"].cuda()

            optimizer1.zero_grad()
            gra_sharp = 2.0

            rPPG, _, _, _ = model(inputs, gra_sharp)
            rPPG = (rPPG - torch.mean(rPPG)) / (torch.std(rPPG) + 1e-8)
            loss_rPPG = criterion_Pearson(rPPG, ecg)

            clip_average_HR = clip_average_HR - 40

            fre_loss = 0.0
            kl_loss = 0.0
            train_mae = 0.0
            for bb in range(inputs.shape[0]):
                loss_distribution_kl, fre_loss_temp, train_mae_temp = (
                    TorchLossComputer.cross_entropy_power_spectrum_DLDL_softmax2(
                        rPPG[bb], clip_average_HR[bb], frame_rate[bb], std=1.0
                    )
                )
                fre_loss += fre_loss_temp
                kl_loss += loss_distribution_kl
                train_mae += train_mae_temp
            fre_loss /= inputs.shape[0]
            kl_loss /= inputs.shape[0]
            train_mae /= inputs.shape[0]

            if epoch > 25:
                a, b = 0.05, 5.0
            else:
                a = a_start * math.pow(exp_a, epoch / 25.0)
                b = b_start * math.pow(exp_b, epoch / 25.0)
            a = 0.1

            loss = a * loss_rPPG + b * (fre_loss + kl_loss)
            loss.backward()
            optimizer1.step()

            n = inputs.size(0)
            loss_rPPG_avg.update(loss_rPPG.detach().item(), n)
            loss_peak_avg.update(fre_loss.detach().item(), n)
            loss_kl_avg_test.update(kl_loss.detach().item(), n)
            loss_hr_mae.update(
                train_mae.detach().item()
                if hasattr(train_mae, "detach")
                else float(train_mae),
                n,
            )

            if i % echo_batches == echo_batches - 1:
                msg = (
                    "epoch:%d, batch:%3d, lr=%f, NegPearson=%.4f, kl=%.4f, fre=%.4f, hr_mae=%.4f"
                    % (
                        epoch + 1,
                        i + 1,
                        lr,
                        loss_rPPG_avg.avg,
                        loss_kl_avg_test.avg,
                        loss_peak_avg.avg,
                        loss_hr_mae.avg,
                    )
                )
                print(msg)
                log_file.write(msg + "\n")
                log_file.flush()
                y1 = 2 * rPPG[0].detach().cpu().numpy()
                y2 = ecg[0].detach().cpu().numpy()
                sio.savemat(
                    os.path.join(args.log, "rPPG.mat"),
                    {"results_rPPG": [y1, y2]},
                )

        # quick validation MAE (HR index) on val manifest
        if val_manifest and n_flat_val > 0:
            model.eval()
            val_dl = UBFC_train(val_manifest, transform=val_tf)
            val_loader = DataLoader(
                val_dl,
                batch_size=min(4, args.batchsize),
                shuffle=False,
                num_workers=args.num_workers,
            )
            val_mae_run = AvgrageMeter()
            with torch.no_grad():
                for vb in val_loader:
                    vi = vb["video_x"].cuda()
                    v_hr = vb["clip_average_HR"].cuda()
                    vf = vb["frame_rate"].cuda()
                    vr, _, _, _ = model(vi, 2.0)
                    vr = (vr - torch.mean(vr)) / (torch.std(vr) + 1e-8)
                    for bb in range(vi.shape[0]):
                        _, _, mae_t = TorchLossComputer.cross_entropy_power_spectrum_DLDL_softmax2(
                            vr[bb], v_hr[bb] - 40.0, vf[bb], std=1.0
                        )
                        val_mae_run.update(mae_t.detach().item(), 1)
            vmsg = "epoch %d val_hr_mae %.4f" % (epoch + 1, val_mae_run.avg)
            print(vmsg)
            log_file.write(vmsg + "\n")
            log_file.flush()
            model.train()

        ckpt_name = os.path.join(
            args.log, "Physformer_UBFC_%d_%d.pkl" % (fold_index, epoch)
        )
        torch.save(model.state_dict(), ckpt_name)

    print("Finished Training")
    log_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PhysFormer train on UBFC-rPPG")
    parser.add_argument(
        "--ubfc_root",
        type=str,
        required=True,
        help="Folder containing subject*/vid.avi and ground_truth.txt",
    )
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--batchsize", type=int, default=4)
    parser.add_argument("--step_size", type=int, default=50)
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--echo_batches", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument(
        "--log",
        type=str,
        default="Physformer_UBFC_run1",
        help="Directory for logs and checkpoints",
    )
    parser.add_argument(
        "--clip_stride",
        type=int,
        default=80,
        help="Frames between clip starts (smaller = more clips)",
    )
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="DataLoader workers (0 recommended on Windows)",
    )
    parser.add_argument(
        "--pretrained",
        type=str,
        default="",
        help="Optional PhysFormer .pkl (e.g. VIPL fold1) for fine-tuning",
    )
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    train_ubfc()
