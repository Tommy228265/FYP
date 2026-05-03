# UBFC-rPPG loader for PhysFormer training (clip length 160, face 128x128).
# Expected layout (rPPG-Toolbox style):
#   UBFC_ROOT/
#     subject1/vid.avi  ground_truth.txt
#     subject2/...
from __future__ import print_function, division

import os
import random

import cv2
import numpy as np
from torch.utils.data import Dataset

clip_frames = 160


def _find_video(subject_dir):
    for name in ("vid.avi", "VID.avi", "vid.mp4", "video.avi", "Vid.avi"):
        p = os.path.join(subject_dir, name)
        if os.path.isfile(p):
            return p
    return None


def _resample_1d(sig, n_out):
    sig = np.asarray(sig, dtype=np.float64).flatten()
    if sig.size == n_out:
        return sig
    if sig.size == 1:
        return np.full(n_out, float(sig[0]))
    x_old = np.linspace(0.0, 1.0, sig.size)
    x_new = np.linspace(0.0, 1.0, n_out)
    return np.interp(x_new, x_old, sig)


def _pseudo_ppg_from_hr(hr_bpm_160, fps):
    """Build a 160-sample pseudo waveform from instantaneous HR (BPM)."""
    hr = np.clip(np.asarray(hr_bpm_160, dtype=np.float64), 40.0, 200.0)
    phase = np.zeros(clip_frames, dtype=np.float64)
    for i in range(1, clip_frames):
        phase[i] = phase[i - 1] + (2.0 * np.pi * (hr[i] / 60.0)) / fps
    return np.sin(phase)


def _estimate_hr_bpm_clip(wave_160, fps=30.0):
    """Dominant frequency in cardiac band -> BPM for DLDL target."""
    x = wave_160 - np.mean(wave_160)
    spec = np.abs(np.fft.rfft(x))
    freqs = np.fft.rfftfreq(clip_frames, d=1.0 / fps)
    band = (freqs >= 0.65) & (freqs <= 3.5)
    if not np.any(band):
        return 70.0
    sub = spec[band]
    freqs_b = freqs[band]
    k = int(np.argmax(sub))
    bpm = float(freqs_b[k] * 60.0)
    return float(np.clip(bpm, 40.0, 180.0))


def _parse_ground_truth(gt_path, n_frames_video, fps):
    """
    Build per-frame waveform (length n_frames_video).

    Official UBFC DATASET_2 `ground_truth.txt` (see authors' sample code):
      row 0: contact PPG waveform (gt_trace)
      row 1: sensor HR (optional)
      row 2: time stamps (gt_time), same length as row 0

    We interpolate row 0 onto video frame times (n_frames_video @ fps).
    Other layouts: 1D long vector, or 2-row heuristics (indices + HR).
    """
    raw = np.loadtxt(gt_path)
    fps = float(fps)

    # --- Official DATASET_2 three-row layout (priority) ---
    if raw.ndim == 2 and raw.shape[0] >= 3:
        gt_trace = raw[0, :].astype(np.float64).flatten()
        gt_time = raw[2, :].astype(np.float64).flatten()
        if gt_trace.size != gt_time.size:
            wave_full = _resample_1d(gt_trace, n_frames_video)
        else:
            order = np.argsort(gt_time)
            t_s = gt_time[order]
            s_s = gt_trace[order]
            t_frames = (np.arange(n_frames_video, dtype=np.float64) + 0.5) / fps
            wave_full = np.interp(
                t_frames, t_s, s_s, left=s_s[0], right=s_s[-1]
            )
    elif raw.ndim == 1:
        sig = raw.astype(np.float64)
        wave_full = _resample_1d(sig, n_frames_video)
    else:
        row2 = raw[1].astype(np.float64)
        med = float(np.nanmedian(row2))
        if 25.0 < med < 220.0:
            hr_full = _resample_1d(row2, n_frames_video)
            phase = 0.0
            wave_full = np.zeros(n_frames_video, dtype=np.float64)
            for i in range(n_frames_video):
                phase = phase + (2.0 * np.pi * (float(hr_full[i]) / 60.0)) / fps
                wave_full[i] = np.sin(phase)
        else:
            wave_full = _resample_1d(row2, n_frames_video)

    wave_full = wave_full - np.mean(wave_full)
    wave_full = wave_full / (np.std(wave_full) + 1e-8)
    return wave_full


class UBFC_train(Dataset):
    """
    Training clips from UBFC-rPPG: random clip each __getitem__ from manifest entry
    or fixed start (manifest stores subject_dir + list of valid starts).

    manifest rows: dict with keys subject_dir, starts (list of int frame indices).
    """

    def __init__(self, manifest, transform=None, cascade_path=None):
        self.manifest = manifest
        self.transform = transform
        path = cascade_path or (
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self._face = cv2.CascadeClassifier(path)
        flat = []
        for m in manifest:
            subj = m["subject_dir"]
            fps = m["fps"]
            wave = m["wave_full"]
            for st in m["starts"]:
                flat.append((subj, st, fps, wave))
        self._flat = flat

    def __len__(self):
        return len(self._flat)

    def __getitem__(self, idx):
        subject_dir, start_frame, fps, wave_full = self._flat[idx]
        vid_path = _find_video(subject_dir)
        if vid_path is None:
            raise FileNotFoundError("No vid.avi/vid.mp4 in " + subject_dir)

        cap = cv2.VideoCapture(vid_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        video_x = np.zeros((clip_frames, 128, 128, 3), dtype=np.float32)
        size_crop = random.randint(0, 15)

        for i in range(clip_frames):
            ret, frame = cap.read()
            if not ret or frame is None:
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
            face = self._crop_face(frame, size_crop)
            video_x[i, :, :, :] = face
        cap.release()

        ecg = wave_full[start_frame : start_frame + clip_frames].astype(np.float64)
        if ecg.shape[0] < clip_frames:
            pad = clip_frames - ecg.shape[0]
            ecg = np.pad(ecg, (0, pad), mode="edge")
        ecg = ecg[:clip_frames]
        ecg = ecg - np.mean(ecg)
        ecg = ecg / (np.std(ecg) + 1e-8)

        clip_average_HR = _estimate_hr_bpm_clip(ecg.astype(np.float64), fps)
        frame_rate = float(fps)

        sample = {
            "video_x": video_x,
            "frame_rate": frame_rate,
            "ecg": ecg.astype(np.float32),
            "clip_average_HR": float(clip_average_HR),
        }
        if self.transform:
            sample = self.transform(sample)
        return sample

    def _crop_face(self, frame, size_crop):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self._face.detectMultiScale(gray, 1.2, 4, minSize=(60, 60))
        h, w = frame.shape[:2]
        if len(faces) == 0:
            y1, y2 = int(h * 0.12), int(h * 0.92)
            x1, x2 = int(w * 0.2), int(w * 0.8)
            roi = frame[y1:y2, x1:x2]
        else:
            x, y, fw, fh = max(faces, key=lambda f: f[2] * f[3])
            pad = int(0.12 * max(fw, fh))
            x0 = max(x - pad, 0)
            y0 = max(y - pad, 0)
            x1 = min(x + fw + pad, w)
            y1 = min(y + fh + pad, h)
            roi = frame[y0:y1, x0:x1]
        ch, cw = roi.shape[0], roi.shape[1]
        if ch < 2 or cw < 2:
            roi = frame
        sc = 128 + size_crop
        resized = cv2.resize(roi, (sc, sc), interpolation=cv2.INTER_CUBIC)
        off = size_crop // 2
        out = resized[off : off + 128, off : off + 128, :]
        return out


def discover_ubfc_subjects(ubfc_root):
    """Return list of subject directories that contain ground_truth.txt."""
    subjs = []
    for dirpath, _, filenames in os.walk(ubfc_root):
        if "ground_truth.txt" in filenames:
            subjs.append(dirpath)
    subjs.sort()
    return subjs


def build_manifest_for_subjects(
    subject_dirs, clip_stride=80, min_tail=0, shuffle_starts=True, rng_seed=0
):
    """
    Build manifest list for UBFC_train.
    clip_stride: step between consecutive clip starts (smaller = more clips, slower epoch).
    """
    rng = random.Random(rng_seed)
    manifest = []
    for subj in subject_dirs:
        gt_path = os.path.join(subj, "ground_truth.txt")
        vid_path = _find_video(subj)
        if not vid_path:
            continue
        cap = cv2.VideoCapture(vid_path)
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cap.release()
        if n_frames < clip_frames + min_tail:
            continue

        wave_full = _parse_ground_truth(gt_path, n_frames, fps)
        if wave_full.shape[0] != n_frames:
            wave_full = _resample_1d(wave_full, n_frames)

        starts = list(range(0, n_frames - clip_frames + 1, clip_stride))
        if shuffle_starts:
            rng.shuffle(starts)
        manifest.append(
            {
                "subject_dir": subj,
                "fps": float(fps),
                "wave_full": wave_full.astype(np.float32),
                "starts": starts,
            }
        )
    return manifest


def split_subject_dirs(subject_dirs, val_ratio=0.2, seed=42):
    rng = random.Random(seed)
    dirs = list(subject_dirs)
    rng.shuffle(dirs)
    if len(dirs) == 1:
        return dirs, dirs
    n_val = max(1, int(len(dirs) * val_ratio))
    if n_val >= len(dirs):
        n_val = 1
    val_set = set(dirs[:n_val])
    train = [d for d in dirs if d not in val_set]
    val = [d for d in dirs if d in val_set]
    return train, val
