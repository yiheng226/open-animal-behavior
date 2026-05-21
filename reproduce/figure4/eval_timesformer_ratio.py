"""
eval_timesformer_ratio.py — Evaluate a trained TimeSformer (ViT-Base) model (ratio variant)

Usage:
    python eval_timesformer_ratio.py --model_path checkpoints/timesformer_ratio/model.pth --test_folds 1
"""

import os
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm.auto import tqdm
import decord
from decord import VideoReader, cpu
from torch.cuda.amp import autocast
from sklearn.metrics import f1_score, confusion_matrix, average_precision_score
from torchvision.transforms import ToTensor
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
SELECTED_BEHAVIORS = ["Aggression", "Investigation", "Allo-groom", "Standing", "Other"]
TIMESFORMER_NUM_FRAMES = 8


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate TimeSformer model (ratio variant)")
    p.add_argument("--model_path", type=str, required=True)
    p.add_argument("--test_folds", nargs="+", type=int, default=[1])
    p.add_argument("--base_video_dir", type=str, default="data/videos")
    p.add_argument("--label_dir", type=str, default="data/labels")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--window_size", type=int, default=16)
    p.add_argument("--stride", type=int, default=4)
    p.add_argument("--skip", type=int, default=0)
    p.add_argument("--mlp_hidden_dim", type=int, default=512)
    p.add_argument("--mlp_dropout", type=float, default=0.3)
    p.add_argument("--smooth_window_size", type=int, default=1)
    p.add_argument("--hf_model", type=str, default="facebook/timesformer-base-finetuned-k400")
    p.add_argument("--save_cm", type=str, default=None)
    return p.parse_args()


# ============================== Utilities ==============================

def build_label_mapping(selected):
    o2n, n2o = {}, {}
    ni = 0
    for oi, name in enumerate(ALL_BEHAVIOR_NAMES):
        if name in selected:
            o2n[oi] = ni
            n2o[ni] = oi
            ni += 1
        else:
            o2n[oi] = None
    return o2n, n2o


def list_videos_in_folders(base_dir, folder_list):
    paths = []
    for folder in folder_list:
        d = os.path.join(base_dir, str(folder))
        if not os.path.isdir(d):
            print(f"[WARN] Folder not found: {d}")
            continue
        for n in sorted(os.listdir(d)):
            if n.lower().endswith(".mp4"):
                paths.append(os.path.join(d, n))
    return paths


def paths_to_labels(video_paths, label_dir):
    lps, kept = [], []
    for vp in video_paths:
        lp = os.path.join(label_dir, os.path.splitext(os.path.basename(vp))[0] + ".csv")
        if os.path.exists(lp):
            kept.append(vp)
            lps.append(lp)
        else:
            print(f"[WARN] Label not found: {os.path.basename(vp)}")
    return kept, lps


def filter_and_remap_labels(labels_oh, o2n):
    orig = np.argmax(labels_oh, axis=1)
    remap = np.full(len(orig), -1, dtype=np.int64)
    valid = np.zeros(len(orig), dtype=bool)
    for i, o in enumerate(orig):
        n = o2n.get(o)
        if n is not None:
            remap[i] = n
            valid[i] = True
    return remap, valid


def uniform_sample_frames(frames, target):
    n = len(frames)
    if n == target:
        return frames
    if n < target:
        return frames + [frames[-1]] * (target - n)
    indices = np.linspace(0, n - 1, target, dtype=int)
    return [frames[i] for i in indices]


def custom_video_transform_timesformer(frames):
    if len(frames) != TIMESFORMER_NUM_FRAMES:
        frames = uniform_sample_frames(frames, TIMESFORMER_NUM_FRAMES)
    frames = [ToTensor()(f) for f in frames]
    video = torch.stack(frames, dim=0)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, -1, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, -1, 1, 1)
    video = (video - mean) / std
    return video.permute(1, 0, 2, 3)


def plot_confusion_matrix(cm, names, path, title="Confusion Matrix"):
    fig, ax = plt.subplots(figsize=(8, 6))
    pct = cm.astype("float") / cm.sum(axis=1, keepdims=True) * 100
    pct = np.nan_to_num(pct)
    sns.heatmap(pct, annot=True, fmt=".1f", cmap="Blues",
                xticklabels=names, yticklabels=names, ax=ax, vmin=0, vmax=100)
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j + 0.5, i + 0.72, f"({cm[i, j]})",
                    ha="center", va="center", fontsize=7, color="gray")
    ax.set_ylabel("True Label")
    ax.set_xlabel("Predicted Label")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"📊 Confusion matrix saved: {path}")


def print_confusion_matrix_text(cm, names):
    print("\n" + "=" * 80)
    print("CONFUSION MATRIX")
    print("=" * 80)
    cw = max(max(len(n) for n in names) + 2, 12)
    print(" " * cw + "Predicted →")
    print(" " * cw + "".join(f"{n:>{cw}}" for n in names))
    print("Actual ↓" + " " * (cw - 8) + "-" * (cw * len(names)))
    for i, n in enumerate(names):
        print(f"{n:>{cw}}" + "".join(f"{cm[i, j]:>{cw}}" for j in range(len(names))))
    print("=" * 80 + "\n")


# ============================== Dataset ==============================

class WindowPredictionDataset(Dataset):
    def __init__(self, video_paths, label_paths, window_size, stride, transform,
                 skip, o2n, num_behaviors, num_classes):
        self.video_paths = video_paths
        self.label_paths = label_paths
        self.window_size = window_size
        self.stride = stride
        self.transform = transform
        self.skip = skip
        self.o2n = o2n
        self.num_behaviors = num_behaviors
        self.num_classes = num_classes
        self.windows, self.frame_mappings = self._generate_windows()

    def _generate_windows(self):
        windows, mappings = [], []
        for vp, lp in zip(self.video_paths, self.label_paths):
            df = pd.read_csv(lp)
            oh = df.iloc[:, 0:self.num_behaviors].values
            vr = VideoReader(vp, ctx=cpu(0))
            T = len(vr)
            if len(oh) != T:
                continue
            remap, valid = filter_and_remap_labels(oh, self.o2n)
            sel = list(range(0, T, self.skip + 1))
            sv = [i for i in sel if i < len(valid) and valid[i]]
            if len(sv) < self.window_size:
                continue
            sl = np.zeros((len(sv), self.num_classes))
            for i, fi in enumerate(sv):
                sl[i, remap[fi]] = 1.0
            f2w = [[] for _ in range(len(sv))]
            for i in range(len(sv) - self.window_size + 1):
                if i % self.stride != 0:
                    continue
                wi = sv[i:i + self.window_size]
                windows.append((vp, wi))
                widx = len(windows) - 1
                for fi in range(i, i + self.window_size):
                    if fi < len(f2w):
                        f2w[fi].append(widx)
            mappings.append({"labels": sl, "frame_to_windows": f2w})
        return windows, mappings

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        vp, fi = self.windows[idx]
        vr = VideoReader(vp, ctx=cpu(0))
        frames = [Image.fromarray(f) for f in vr.get_batch(fi).asnumpy()]
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
        x = self.norm(x)
        return self.fc2(self.dropout(self.relu(self.fc1(x))))


class CustomTimeSformer(nn.Module):
    def __init__(self, num_classes, hf_model, hidden_dim=512, dropout=0.3):
        super().__init__()
        from transformers import TimesformerModel
        self.backbone = TimesformerModel.from_pretrained(hf_model)
        hs = self.backbone.config.hidden_size
        self.head = MLPHead(hs, num_classes, hidden_dim, dropout)

    def forward(self, x):
        x = x.permute(0, 2, 1, 3, 4)
        out = self.backbone(pixel_values=x)
        return self.head(out.last_hidden_state[:, 0])


def build_timesformer(num_classes, hf_model, hidden_dim=512, dropout=0.3):
    return CustomTimeSformer(num_classes, hf_model, hidden_dim, dropout)


# ============================== Evaluation ==============================

def evaluate_framewise(model, loader, mappings, num_classes, smooth_k, device):
    model.eval()
    wp = []
    with torch.no_grad():
        for vids, _ in tqdm(loader, desc="Evaluating", leave=False):
            vids = vids.to(device)
            with autocast():
                wp.extend(torch.softmax(model(vids), dim=1).cpu().numpy())
    wp = np.array(wp)

    all_labels, all_fp = [], []
    for m in mappings:
        labels, f2w = m["labels"], m["frame_to_windows"]
        F = len(labels)
        fp = np.full((F, num_classes), 1.0 / num_classes, dtype=np.float32)
        for f in range(F):
            if f2w[f]:
                fp[f] = np.mean(wp[f2w[f]], axis=0)
        all_fp.append(fp)
        all_labels.extend(np.argmax(labels, axis=1))

    def smooth(p, k):
        if k <= 1:
            return p
        h = k // 2
        o = np.zeros_like(p)
        for i in range(len(p)):
            o[i] = np.mean(p[max(0, i - h):min(len(p), i + h + 1)], axis=0)
        return o

    preds, raw = [], []
    for fp in all_fp:
        raw.extend(fp.tolist())
        preds.extend(np.argmax(smooth(fp, smooth_k), axis=1).tolist())

    lbls = list(range(num_classes))
    f1_pc = f1_score(all_labels, preds, average=None, labels=lbls)
    f1_m = f1_score(all_labels, preds, average="macro")
    cm = confusion_matrix(all_labels, preds, labels=lbls)

    oh = np.zeros((len(all_labels), num_classes))
    for i, l in enumerate(all_labels):
        oh[i, l] = 1.0
    raw = np.array(raw)
    ap_pc = np.array([
        average_precision_score(oh[:, c], raw[:, c]) if oh[:, c].sum() > 0 else float("nan")
        for c in range(num_classes)
    ])

    return {
        "f1_per_class": f1_pc, "f1_macro": f1_m, "cm": cm,
        "ap_per_class": ap_pc, "mAP": np.nanmean(ap_pc),
    }


# ============================== Main ==============================

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    o2n, _ = build_label_mapping(SELECTED_BEHAVIORS)
    num_classes = len(SELECTED_BEHAVIORS)

    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model not found: {args.model_path}")

    print(f"\n{'='*80}")
    print("MODEL EVALUATION — TimeSformer (ViT-Base, K400) (ratio variant)")
    print(f"{'='*80}")
    print(f"  Model: {args.model_path}")
    print(f"  Test folds: {args.test_folds}")
    print(f"  Behaviors: {SELECTED_BEHAVIORS}\n")

    vids = list_videos_in_folders(args.base_video_dir, args.test_folds)
    vids, labs = paths_to_labels(vids, args.label_dir)
    print(f"  Test videos: {len(vids)}\n")

    ds = WindowPredictionDataset(vids, labs, args.window_size, args.stride,
                                 custom_video_transform_timesformer, args.skip,
                                 o2n, len(ALL_BEHAVIOR_NAMES), num_classes)
    loader = DataLoader(ds, args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)
    print(f"  Test windows: {len(ds)}\n")

    print("  Loading model...")
    model = build_timesformer(num_classes, args.hf_model,
                              args.mlp_hidden_dim, args.mlp_dropout).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    print("  ✅ Model loaded\n")

    metrics = evaluate_framewise(model, loader, ds.frame_mappings, num_classes,
                                 args.smooth_window_size, device)

    print(f"{'='*80}")
    print("RESULTS")
    print(f"{'='*80}\n")
    print(f"  F1 Macro:  {metrics['f1_macro']:.4f}")
    print(f"  mAP:       {metrics['mAP']:.4f}\n")
    print("  Per-class F1:")
    for n, v in zip(SELECTED_BEHAVIORS, metrics["f1_per_class"]):
        print(f"    {n:>15}: {v:.4f}")
    print("\n  Per-class AP:")
    for n, v in zip(SELECTED_BEHAVIORS, metrics["ap_per_class"]):
        print(f"    {n:>15}: {v:.4f}")

    print_confusion_matrix_text(metrics["cm"], SELECTED_BEHAVIORS)

    if args.save_cm:
        plot_confusion_matrix(metrics["cm"], SELECTED_BEHAVIORS, args.save_cm,
                              f"TimeSformer Eval (F1={metrics['f1_macro']:.4f})")

    print("✅ Evaluation complete!")


if __name__ == "__main__":
    main()
