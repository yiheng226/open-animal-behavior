"""
train_swin3d.py — Video Swin Transformer-T (Kinetics-400 Pretrained) Training
3-Fold Cross-Validation for Mouse Behavior Classification

Usage:
    python train_swin3d.py                          # Default: Fold 1 (train on 2,3 / val on 1)
    python train_swin3d.py --test_folds 2 --train_folds 1 3   # Fold 2
    python train_swin3d.py --test_folds 3 --train_folds 1 2   # Fold 3
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageFilter
from tqdm.auto import tqdm
from collections import Counter
import decord
from decord import VideoReader, cpu
from torch.cuda.amp import GradScaler, autocast
from sklearn.metrics import f1_score, confusion_matrix, average_precision_score
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision.models.video import swin3d_t, Swin3D_T_Weights
from torchvision.transforms import ToTensor
import random
import shutil
from datetime import datetime
import json
import time
import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# ============================== Configuration ==============================

ALL_BEHAVIOR_NAMES = [
    "Aggression", "Investigation", "Allo-groom",
    "Self-groom", "Standing", "Chasing", "Other",
]

DEFAULTS = dict(
    test_folds=[1],
    train_folds=[3, 2],
    base_video_dir="data/videos",
    label_dir="data/labels",
    selected_behaviors=["Aggression", "Investigation", "Allo-groom", "Standing", "Other"],
    batch_size=8,
    accumulation_steps=2,
    num_epochs=5,
    base_lr=3.8e-5,
    weight_decay=0.01,
    num_workers=8,
    use_class_weights=False,
    aug_blur=True,
    aug_blur_frac=0.35,
    aug_td=True,
    aug_td_frac=0.15,
    window_size=16,
    stride=4,
    skip=0,
    T=8,
    mlp_hidden_dim=512,
    mlp_dropout=0.3,
    smooth_window_size=1,
    seed=2025,
    model_save_dir="checkpoints/swin3d",
)


def parse_args():
    p = argparse.ArgumentParser(description="Train Video Swin-T for behavior classification")
    p.add_argument("--test_folds", nargs="+", type=int, default=DEFAULTS["test_folds"])
    p.add_argument("--train_folds", nargs="+", type=int, default=DEFAULTS["train_folds"])
    p.add_argument("--base_video_dir", type=str, default=DEFAULTS["base_video_dir"])
    p.add_argument("--label_dir", type=str, default=DEFAULTS["label_dir"])
    p.add_argument("--batch_size", type=int, default=DEFAULTS["batch_size"])
    p.add_argument("--accumulation_steps", type=int, default=DEFAULTS["accumulation_steps"])
    p.add_argument("--num_epochs", type=int, default=DEFAULTS["num_epochs"])
    p.add_argument("--base_lr", type=float, default=DEFAULTS["base_lr"])
    p.add_argument("--weight_decay", type=float, default=DEFAULTS["weight_decay"])
    p.add_argument("--num_workers", type=int, default=DEFAULTS["num_workers"])
    p.add_argument("--window_size", type=int, default=DEFAULTS["window_size"])
    p.add_argument("--stride", type=int, default=DEFAULTS["stride"])
    p.add_argument("--skip", type=int, default=DEFAULTS["skip"])
    p.add_argument("--mlp_hidden_dim", type=int, default=DEFAULTS["mlp_hidden_dim"])
    p.add_argument("--mlp_dropout", type=float, default=DEFAULTS["mlp_dropout"])
    p.add_argument("--smooth_window_size", type=int, default=DEFAULTS["smooth_window_size"])
    p.add_argument("--seed", type=int, default=DEFAULTS["seed"])
    p.add_argument("--model_save_dir", type=str, default=DEFAULTS["model_save_dir"])
    p.add_argument("--use_class_weights", action="store_true")
    return p.parse_args()


# ============================== Utilities ==============================

def build_label_mapping(selected_behaviors):
    original_to_new = {}
    new_to_original = {}
    new_idx = 0
    for orig_idx, name in enumerate(ALL_BEHAVIOR_NAMES):
        if name in selected_behaviors:
            original_to_new[orig_idx] = new_idx
            new_to_original[new_idx] = orig_idx
            new_idx += 1
        else:
            original_to_new[orig_idx] = None
    return original_to_new, new_to_original


def list_videos_in_folders(base_dir, folder_list):
    video_paths = []
    for folder in folder_list:
        fold_dir = os.path.join(base_dir, str(folder))
        if not os.path.isdir(fold_dir):
            print(f"[WARN] Folder not found: {fold_dir}")
            continue
        for name in sorted(os.listdir(fold_dir)):
            if name.lower().endswith(".mp4"):
                video_paths.append(os.path.join(fold_dir, name))
    return video_paths


def paths_to_labels(video_paths, label_dir):
    label_paths, kept = [], []
    for vp in video_paths:
        csv_name = os.path.splitext(os.path.basename(vp))[0] + ".csv"
        lp = os.path.join(label_dir, csv_name)
        if os.path.exists(lp):
            kept.append(vp)
            label_paths.append(lp)
        else:
            print(f"[WARN] Label not found: {os.path.basename(vp)}")
    return kept, label_paths


def filter_and_remap_labels(labels_onehot, original_to_new):
    original_labels = np.argmax(labels_onehot, axis=1)
    remapped = np.full(len(original_labels), -1, dtype=np.int64)
    valid = np.zeros(len(original_labels), dtype=bool)
    for i, ol in enumerate(original_labels):
        nl = original_to_new.get(ol)
        if nl is not None:
            remapped[i] = nl
            valid[i] = True
    return remapped, valid


def random_blur_frames(frames, frac=0.35, radius_range=(0.8, 2.2), rng=None):
    if frac <= 0 or len(frames) == 0:
        return frames
    rng = rng or random
    n = len(frames)
    k = max(1, int(round(n * frac)))
    idxs = rng.sample(range(n), k)
    rmin, rmax = radius_range
    out = []
    for i, im in enumerate(frames):
        if i in idxs:
            out.append(im.filter(ImageFilter.GaussianBlur(radius=rng.uniform(rmin, rmax))))
        else:
            out.append(im)
    return out


def random_temporal_dropout(frames, frac=0.15, rng=None):
    if frac <= 0 or len(frames) < 3:
        return frames
    rng = rng or random
    n = len(frames)
    k = max(1, int(round(n * frac)))
    idxs = rng.sample(range(1, n - 1), min(k, max(1, n - 2)))
    out = frames[:]
    for i in idxs:
        out[i] = out[i - 1] if rng.random() < 0.5 else out[i + 1]
    return out


def custom_video_transform(frames):
    """Output shape: (C, T, H, W)"""
    frames = [ToTensor()(f) for f in frames]
    video = torch.stack(frames, dim=1)  # (C, T, H, W)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1, 1)
    return (video - mean) / std


def plot_confusion_matrix(cm, class_names, save_path, title="Confusion Matrix"):
    fig, ax = plt.subplots(figsize=(8, 6))
    cm_pct = cm.astype("float") / cm.sum(axis=1, keepdims=True) * 100
    cm_pct = np.nan_to_num(cm_pct)
    sns.heatmap(cm_pct, annot=True, fmt=".1f", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax, vmin=0, vmax=100)
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j + 0.5, i + 0.72, f"({cm[i, j]})",
                    ha="center", va="center", fontsize=7, color="gray")
    ax.set_ylabel("True Label")
    ax.set_xlabel("Predicted Label")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"📊 Confusion matrix saved: {save_path}")


# ============================== Datasets ==============================

class SlidingWindowVideoDataset(Dataset):
    def __init__(self, video_paths, label_paths, window_size, stride, transform,
                 skip=0, augment=None, original_to_new=None, num_behaviors=7):
        self.video_paths = video_paths
        self.label_paths = label_paths
        self.window_size = window_size
        self.stride = stride
        self.transform = transform
        self.skip = skip
        self.augment = augment
        self.original_to_new = original_to_new
        self.num_behaviors = num_behaviors
        self.samples, self.sample_labels = self._generate_samples()

    def _generate_samples(self):
        samples, sample_labels = [], []
        for vp, lp in zip(self.video_paths, self.label_paths):
            df = pd.read_csv(lp)
            labels_oh = df.iloc[:, 0:self.num_behaviors].values
            vr = VideoReader(vp, ctx=cpu(0))
            T = len(vr)
            if len(labels_oh) != T:
                print(f"⚠️ Mismatch: {os.path.basename(vp)}")
                continue
            remapped, valid = filter_and_remap_labels(labels_oh, self.original_to_new)
            selected = list(range(0, T, self.skip + 1))
            selected_valid = [i for i in selected if i < len(valid) and valid[i]]
            if len(selected_valid) < self.window_size:
                continue
            for i in range(len(selected_valid) - self.window_size + 1):
                if i % self.stride != 0:
                    continue
                win_idx = selected_valid[i:i + self.window_size]
                win_label = Counter(remapped[win_idx]).most_common(1)[0][0]
                samples.append((vp, win_idx, win_label))
                sample_labels.append(win_label)
        return samples, sample_labels

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        vp, frame_indices, label = self.samples[idx]
        vr = VideoReader(vp, ctx=cpu(0))
        frames = vr.get_batch(frame_indices).asnumpy()
        frames = [Image.fromarray(f) for f in frames]
        if len(frames) < self.window_size:
            frames.extend([frames[-1]] * (self.window_size - len(frames)))
        if self.augment is not None:
            frames = self.augment(frames)
        return self.transform(frames), torch.tensor(label, dtype=torch.long)


class WindowPredictionDataset(Dataset):
    def __init__(self, video_paths, label_paths, window_size, stride, transform,
                 skip=0, original_to_new=None, num_behaviors=7, num_classes=5):
        self.video_paths = video_paths
        self.label_paths = label_paths
        self.window_size = window_size
        self.stride = stride
        self.transform = transform
        self.skip = skip
        self.original_to_new = original_to_new
        self.num_behaviors = num_behaviors
        self.num_classes = num_classes
        self.windows, self.frame_mappings = self._generate_windows()

    def _generate_windows(self):
        windows, frame_mappings = [], []
        for vp, lp in zip(self.video_paths, self.label_paths):
            df = pd.read_csv(lp)
            labels_oh = df.iloc[:, 0:self.num_behaviors].values
            vr = VideoReader(vp, ctx=cpu(0))
            T = len(vr)
            if len(labels_oh) != T:
                continue
            remapped, valid = filter_and_remap_labels(labels_oh, self.original_to_new)
            selected = list(range(0, T, self.skip + 1))
            selected_valid = [i for i in selected if i < len(valid) and valid[i]]
            if len(selected_valid) < self.window_size:
                continue
            sel_labels_oh = np.zeros((len(selected_valid), self.num_classes))
            for i, fi in enumerate(selected_valid):
                sel_labels_oh[i, remapped[fi]] = 1.0
            f2w = [[] for _ in range(len(selected_valid))]
            for i in range(len(selected_valid) - self.window_size + 1):
                if i % self.stride != 0:
                    continue
                win_idx = selected_valid[i:i + self.window_size]
                windows.append((vp, win_idx))
                widx = len(windows) - 1
                for fi in range(i, i + self.window_size):
                    if fi < len(f2w):
                        f2w[fi].append(widx)
            frame_mappings.append({"labels": sel_labels_oh, "frame_to_windows": f2w})
        return windows, frame_mappings

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        vp, frame_indices = self.windows[idx]
        vr = VideoReader(vp, ctx=cpu(0))
        frames = vr.get_batch(frame_indices).asnumpy()
        frames = [Image.fromarray(f) for f in frames]
        if len(frames) < self.window_size:
            frames.extend([frames[-1]] * (self.window_size - len(frames)))
        return self.transform(frames), idx


# ============================== Model ==============================

class MLPHead(nn.Module):
    def __init__(self, in_features, num_classes, hidden_dim=512, dropout=0.3):
        super().__init__()
        self.norm = nn.LayerNorm(in_features)
        self.fc1 = nn.Linear(in_features, hidden_dim)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = torch.mean(x, dim=1)
        x = self.norm(x)
        return self.fc2(self.dropout(self.relu(self.fc1(x))))


class CustomSwin3D(nn.Module):
    def __init__(self, pretrained=True, T=8):
        super().__init__()
        weights = Swin3D_T_Weights.DEFAULT if pretrained else None
        self.model = swin3d_t(weights=weights)
        self.T = T
        self.model.head = nn.Identity()
        self.model.avgpool = nn.Identity()

    def forward(self, x):
        x = self.model.patch_embed(x)
        x = self.model.pos_drop(x)
        x = self.model.features(x)
        x = self.model.norm(x)
        x = x.mean(dim=(2, 3))  # spatial pooling → (B, T, C)
        return self.head(x)


def build_swin3d(num_classes, pretrained=True, T=8, hidden_dim=512, dropout=0.3):
    model = CustomSwin3D(pretrained=pretrained, T=T)
    model.head = MLPHead(768, num_classes, hidden_dim, dropout)
    return model


# ============================== Evaluation ==============================

def evaluate_framewise(model, loader, frame_mappings, num_classes, smooth_k=1, device="cuda"):
    model.eval()
    window_probs = []
    with torch.no_grad():
        for videos, _ in tqdm(loader, desc="Validation", leave=False):
            videos = videos.to(device)
            with autocast():
                probs = torch.softmax(model(videos), dim=1).cpu().numpy()
            window_probs.extend(probs)
    window_probs = np.array(window_probs)

    all_labels, all_frame_probs = [], []
    for mapping in frame_mappings:
        labels = mapping["labels"]
        f2w = mapping["frame_to_windows"]
        F = len(labels)
        fp = np.full((F, num_classes), 1.0 / num_classes, dtype=np.float32)
        for f in range(F):
            if f2w[f]:
                fp[f] = np.mean(window_probs[f2w[f]], axis=0)
        all_frame_probs.append(fp)
        all_labels.extend(np.argmax(labels, axis=1))

    def smooth(probs, k):
        if k <= 1:
            return probs
        h = k // 2
        out = np.zeros_like(probs)
        for i in range(len(probs)):
            out[i] = np.mean(probs[max(0, i - h):min(len(probs), i + h + 1)], axis=0)
        return out

    preds, raw_probs = [], []
    for fp in all_frame_probs:
        raw_probs.extend(fp.tolist())
        preds.extend(np.argmax(smooth(fp, smooth_k), axis=1).tolist())

    labels_arr = list(range(num_classes))
    f1_pc = f1_score(all_labels, preds, average=None, labels=labels_arr)
    f1_m = f1_score(all_labels, preds, average="macro")
    cm = confusion_matrix(all_labels, preds, labels=labels_arr)

    oh = np.zeros((len(all_labels), num_classes))
    for i, l in enumerate(all_labels):
        oh[i, l] = 1.0
    raw_probs = np.array(raw_probs)
    ap_pc = np.array([
        average_precision_score(oh[:, c], raw_probs[:, c])
        if oh[:, c].sum() > 0 else float("nan")
        for c in range(num_classes)
    ])

    return {
        "f1_per_class": f1_pc,
        "f1_macro": f1_m,
        "cm": cm,
        "ap_per_class": ap_pc,
        "mAP": np.nanmean(ap_pc),
    }


# ============================== Main ==============================

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    selected = DEFAULTS["selected_behaviors"]
    original_to_new, _ = build_label_mapping(selected)
    num_classes = len(selected)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    os.makedirs(args.model_save_dir, exist_ok=True)
    fold_tag = f"swin3d_train_{'_'.join(map(str, args.train_folds))}_val_{'_'.join(map(str, args.test_folds))}"

    print(f"\n{'='*70}")
    print(f"Video Swin-T Training — {fold_tag}")
    print(f"{'='*70}")
    print(f"  Behaviors: {selected}")
    print(f"  Device: {device}")
    print(f"  Epochs: {args.num_epochs}, LR: {args.base_lr:.2e}")
    print(f"{'='*70}\n")

    # Load data
    train_vids = list_videos_in_folders(args.base_video_dir, args.train_folds)
    val_vids = list_videos_in_folders(args.base_video_dir, args.test_folds)
    train_vids, train_labs = paths_to_labels(train_vids, args.label_dir)
    val_vids, val_labs = paths_to_labels(val_vids, args.label_dir)
    print(f"Train: {len(train_vids)} videos | Val: {len(val_vids)} videos\n")

    aug_rng = random.Random(args.seed)
    def augment(frames):
        if DEFAULTS["aug_blur"]:
            frames = random_blur_frames(frames, DEFAULTS["aug_blur_frac"], rng=aug_rng)
        if DEFAULTS["aug_td"]:
            frames = random_temporal_dropout(frames, DEFAULTS["aug_td_frac"], rng=aug_rng)
        return frames

    train_ds = SlidingWindowVideoDataset(
        train_vids, train_labs, args.window_size, args.stride,
        custom_video_transform, args.skip, augment,
        original_to_new, len(ALL_BEHAVIOR_NAMES),
    )
    val_ds = WindowPredictionDataset(
        val_vids, val_labs, args.window_size, args.stride,
        custom_video_transform, args.skip,
        original_to_new, len(ALL_BEHAVIOR_NAMES), num_classes,
    )
    print(f"Train windows: {len(train_ds)} | Val windows: {len(val_ds)}\n")

    counts = Counter(train_ds.sample_labels)
    for c in range(num_classes):
        n = counts.get(c, 0)
        print(f"  {selected[c]:>15}: {n:>6} ({100*n/len(train_ds):>5.2f}%)")
    print()

    train_loader = DataLoader(train_ds, args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    model = build_swin3d(num_classes, pretrained=True, T=DEFAULTS["T"],
                         hidden_dim=args.mlp_hidden_dim, dropout=args.mlp_dropout).to(device)

    if args.use_class_weights:
        w = np.array([counts.get(i, 1) for i in range(num_classes)], dtype=np.float32)
        w = w.sum() / (num_classes * w)
        w /= w.mean()
        criterion = nn.CrossEntropyLoss(weight=torch.tensor(w).to(device))
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = optim.AdamW(model.parameters(), lr=args.base_lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.num_epochs)
    scaler = GradScaler()

    log, best_f1 = [], -1.0

    for epoch in range(args.num_epochs):
        t0 = time.time()
        model.train()
        running_loss = 0.0
        optimizer.zero_grad(set_to_none=True)

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.num_epochs}")
        for bi, (vids, tgts) in enumerate(pbar):
            vids, tgts = vids.to(device), tgts.to(device)
            with autocast():
                loss = criterion(model(vids), tgts) / args.accumulation_steps
            scaler.scale(loss).backward()
            if (bi + 1) % args.accumulation_steps == 0 or (bi + 1) == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            running_loss += loss.item() * args.accumulation_steps * vids.size(0)
            pbar.set_postfix(loss=loss.item() * args.accumulation_steps)

        scheduler.step()
        train_time = time.time() - t0
        epoch_loss = running_loss / len(train_ds)

        t1 = time.time()
        metrics = evaluate_framewise(model, val_loader, val_ds.frame_mappings,
                                     num_classes, args.smooth_window_size, device)
        val_time = time.time() - t1

        f1, mAP = metrics["f1_macro"], metrics["mAP"]
        print(f"\nEpoch {epoch+1} | Loss: {epoch_loss:.4f} | F1: {f1:.4f} | mAP: {mAP:.4f}")
        for name, v in zip(selected, metrics["f1_per_class"]):
            print(f"  {name:>15} F1: {v:.4f}")
        for name, v in zip(selected, metrics["ap_per_class"]):
            print(f"  {name:>15} AP: {v:.4f}")
        print(f"  Train: {train_time:.1f}s | Val: {val_time:.1f}s")

        cm_path = os.path.join(args.model_save_dir, f"{fold_tag}_ep{epoch+1}_cm.png")
        plot_confusion_matrix(metrics["cm"], selected, cm_path,
                              f"Swin3D Epoch {epoch+1} (F1={f1:.4f})")

        ckpt = os.path.join(args.model_save_dir,
                            f"{fold_tag}_ep{epoch+1}_f1_{f1:.4f}_map_{mAP:.4f}.pth")
        torch.save(model.state_dict(), ckpt)
        print(f"💾 {ckpt}")

        if f1 > best_f1:
            best_f1 = f1

        log.append({
            "epoch": epoch + 1, "train_loss": epoch_loss,
            "val_f1": f1, "val_map": mAP,
            "val_f1_per_class": metrics["f1_per_class"].tolist(),
            "val_ap_per_class": metrics["ap_per_class"].tolist(),
            "confusion_matrix": metrics["cm"].tolist(),
            "train_time": train_time, "val_time": val_time,
        })
        print()

    best = max(log, key=lambda x: x["val_f1"])
    print(f"\n✅ Best: Epoch {best['epoch']} | F1={best['val_f1']:.4f} | mAP={best['val_map']:.4f}")

    log_path = os.path.join(args.model_save_dir, f"training_log_{fold_tag}.json")
    with open(log_path, "w") as f:
        json.dump({"training_log": log, "best_epoch": best, "config": vars(args)}, f, indent=2)
    print(f"💾 Log: {log_path}\n")


if __name__ == "__main__":
    main()
