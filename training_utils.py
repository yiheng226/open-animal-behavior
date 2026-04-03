"""
Training utilities for animal behavior classification.

Includes dataset classes, augmentation, label mapping,
frame-wise evaluation metrics, and the training loop.
"""

import json
import os
import random
from collections import Counter
from difflib import SequenceMatcher

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from decord import VideoReader, cpu
from PIL import Image, ImageFilter
from sklearn.metrics import average_precision_score, f1_score
from sklearn.model_selection import train_test_split
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset

from inference import preprocess, uniform_sample


# ====================== Augmentation ======================

def random_blur(frames, frac=0.35, rng=None):
    if frac <= 0 or not frames:
        return frames
    rng = rng or random
    n = len(frames)
    k = max(1, int(round(n * frac)))
    idxs = set(rng.sample(range(n), k))
    return [
        f.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.8, 2.2))) if i in idxs else f
        for i, f in enumerate(frames)
    ]


def temporal_dropout(frames, frac=0.15, rng=None):
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


# ====================== Datasets ======================

class SlidingWindowDataset(Dataset):
    """Training dataset using sliding windows over labeled videos."""

    def __init__(self, video_paths, label_paths, ws, stride, cfg, nc, label_map, skip=0, augment=None):
        self.cfg = cfg
        self.ws = ws
        self.augment = augment
        self.samples = []
        self.sample_labels = []

        for vp, lp in zip(video_paths, label_paths):
            try:
                df = pd.read_csv(lp)
                oh = df.values
                vr = VideoReader(vp, ctx=cpu(0))
                T = len(vr)
                n_cols = oh.shape[1]
                if T != len(oh):
                    print(f"⚠️ Length mismatch {vp}")
                    continue
                raw = np.argmax(oh[:, :n_cols], axis=1)
                mapped = np.array([
                    label_map.get(int(l), -1) if label_map.get(int(l)) is not None else -1
                    for l in raw
                ])
                sel = list(range(0, T, skip + 1))
                valid = [i for i in sel if i < len(mapped) and mapped[i] >= 0]
                if len(valid) < ws:
                    continue
                for s in range(0, len(valid) - ws + 1, stride):
                    idx = valid[s:s + ws]
                    lbl = Counter(mapped[idx]).most_common(1)[0][0]
                    self.samples.append((vp, idx, int(lbl)))
                    self.sample_labels.append(int(lbl))
            except Exception as e:
                print(f"⚠️ Skipped {vp}: {e}")
                continue

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        vp, idx, lbl = self.samples[i]
        vr = VideoReader(vp, ctx=cpu(0))
        frames = [Image.fromarray(f) for f in vr.get_batch(idx).asnumpy()]
        if len(frames) < self.ws:
            frames += [frames[-1]] * (self.ws - len(frames))
        if self.augment:
            frames = self.augment(frames)
        return preprocess(frames, self.cfg), torch.tensor(lbl, dtype=torch.long)


class WindowPredictionDataset(Dataset):
    """
    Frame-wise evaluation dataset.

    Generates overlapping windows and tracks which windows cover each frame,
    enabling per-frame probability aggregation for evaluation.
    """

    def __init__(self, video_paths, label_paths, ws, stride, cfg, nc, label_map, skip=0):
        self.cfg = cfg
        self.ws = ws
        self.windows = []
        self.frame_mappings = []

        for vp, lp in zip(video_paths, label_paths):
            try:
                df = pd.read_csv(lp)
                oh = df.values
                vr = VideoReader(vp, ctx=cpu(0))
                T = len(vr)
                n_cols = oh.shape[1]
                if T != len(oh):
                    print(f"⚠️ Length mismatch {vp}")
                    continue
                raw = np.argmax(oh[:, :n_cols], axis=1)
                mapped = np.array([
                    label_map.get(int(l)) if label_map.get(int(l)) is not None else -1
                    for l in raw
                ])
                sel = list(range(0, T, skip + 1))
                valid = [i for i in sel if i < len(mapped) and mapped[i] >= 0]
                if len(valid) < ws:
                    continue
                sel_labels_oh = np.zeros((len(valid), nc), dtype=np.float32)
                for i, fi in enumerate(valid):
                    sel_labels_oh[i, mapped[fi]] = 1.0
                frame_to_windows = [[] for _ in range(len(valid))]
                for i in range(0, len(valid) - ws + 1, stride):
                    win = valid[i:i + ws]
                    self.windows.append((vp, win))
                    widx = len(self.windows) - 1
                    for fi in range(i, i + ws):
                        if fi < len(frame_to_windows):
                            frame_to_windows[fi].append(widx)
                self.frame_mappings.append({
                    "labels": sel_labels_oh,
                    "frame_to_windows": frame_to_windows,
                })
            except Exception as e:
                print(f"⚠️ Skipped {vp}: {e}")
                continue

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, i):
        vp, idx = self.windows[i]
        vr = VideoReader(vp, ctx=cpu(0))
        frames = [Image.fromarray(f) for f in vr.get_batch(idx).asnumpy()]
        if len(frames) < self.ws:
            frames += [frames[-1]] * (self.ws - len(frames))
        return preprocess(frames, self.cfg), i


# ====================== Frame-wise Metrics ======================

def compute_framewise_metrics(window_probs, frame_mappings, nc, smooth_k=1):
    """
    Compute frame-wise F1-macro, mAP, and per-class metrics.

    Args:
        window_probs: np.ndarray of shape (num_windows, nc)
        frame_mappings: list of dicts with 'labels' and 'frame_to_windows'
        nc: number of classes
        smooth_k: temporal smoothing kernel size (1 = no smoothing)

    Returns:
        dict with keys: f1, mAP, f1_per, ap_per
    """
    all_labels = []
    all_frame_probs = []

    for mapping in frame_mappings:
        sel_labels = mapping["labels"]
        f2w = mapping["frame_to_windows"]
        F = len(sel_labels)
        fp = np.zeros((F, nc), dtype=np.float32)
        for f in range(F):
            if f2w[f]:
                fp[f] = np.mean(window_probs[f2w[f]], axis=0)
            else:
                fp[f, :] = 1.0 / nc
        all_frame_probs.append(fp)
        all_labels.extend(np.argmax(sel_labels, axis=1).tolist())

    def smooth(probs, k):
        if k <= 1:
            return probs
        half = k // 2
        out = np.zeros_like(probs)
        for i in range(len(probs)):
            s, e = max(0, i - half), min(len(probs), i + half + 1)
            out[i] = np.mean(probs[s:e], axis=0)
        return out

    preds = []
    probs_raw = []
    for fp in all_frame_probs:
        probs_raw.extend(fp.tolist())
        preds.extend(np.argmax(smooth(fp, smooth_k), axis=1).tolist())

    f1p = f1_score(all_labels, preds, average=None, labels=list(range(nc)), zero_division=0).tolist()
    f1m = f1_score(all_labels, preds, average="macro", zero_division=0)

    probs_arr = np.array(probs_raw)
    oh = np.zeros((len(all_labels), nc))
    for i, l in enumerate(all_labels):
        oh[i, l] = 1

    app = []
    for ci in range(nc):
        try:
            app.append(average_precision_score(oh[:, ci], probs_arr[:, ci]))
        except Exception:
            app.append(0.0)
    mAP = float(np.mean(app))

    return {"f1": f1m, "mAP": mAP, "f1_per": f1p, "ap_per": app}


# ====================== Label Mapping ======================

def fuzzy_match(data_name, pretrained_names):
    """Fuzzy-match a data class name to the closest pretrained class name."""
    dn = data_name.lower().replace("_", " ")
    best_score, best_match = 0, None
    for pn in pretrained_names:
        pnl = pn.lower().replace("_", " ")
        if dn == pnl:
            return pn
        score = SequenceMatcher(None, dn, pnl).ratio()
        if dn in pnl or pnl in dn:
            score = max(score, 0.75)
        if score > best_score:
            best_score = score
            best_match = pn
    return best_match if best_score >= 0.55 else None


def compute_label_map(mode, dd_values, data_labels, pretrained_names):
    """
    Compute (new_class_names, label_map) from user mapping selections.

    Args:
        mode: "Pretrain head" or "New head"
        dd_values: list of selections (one per data class)
        data_labels: original class names from data CSV headers
        pretrained_names: class names from the pretrained model config

    Returns:
        (new_names, label_map)
        label_map maps original index → new index (None = exclude)
    """
    N = len(data_labels)

    if mode == "Pretrain head":
        mapping = {}
        other_list = []
        exclude_list = []
        for i in range(N):
            v = dd_values[i] if i < len(dd_values) else "→ Other"
            if v == "→ Exclude":
                exclude_list.append(i)
            elif v == "→ Other":
                other_list.append(i)
            else:
                mapping[i] = v
        for i in other_list:
            mapping[i] = "others"
        used = set(mapping.values())
        new_names = [n for n in pretrained_names if n in used]
        if "others" in used and "others" not in new_names:
            new_names.append("others")
        label_map = {}
        for i in range(N):
            if i in exclude_list:
                label_map[i] = None
            elif i in mapping:
                label_map[i] = new_names.index(mapping[i])
        return new_names, label_map

    # New head mode
    kept_set = set()
    merge_targets = {}
    other_list = []
    exclude_list = []
    for i in range(N):
        v = dd_values[i] if i < len(dd_values) else "keep"
        if "(keep)" in str(v) or v == "keep":
            kept_set.add(data_labels[i])
        elif "merge into" in str(v):
            merge_targets[i] = v.replace("→ merge into ", "")
        elif v == "→ Other":
            other_list.append(i)
        elif v == "→ Exclude":
            exclude_list.append(i)

    new_names = [nm for nm in data_labels if nm in kept_set]
    if other_list:
        has_o = any(c.lower() in ("other", "others") for c in new_names)
        if not has_o:
            new_names.append("Other")

    label_map = {}
    for i in range(N):
        if i in exclude_list:
            label_map[i] = None
            continue
        nm = data_labels[i]
        if nm in new_names:
            label_map[i] = new_names.index(nm)
        elif i in merge_targets:
            t = merge_targets[i]
            label_map[i] = new_names.index(t) if t in new_names else (
                new_names.index("Other") if "Other" in new_names else 0
            )
        elif i in other_list:
            oidx = next(
                (j for j, n in enumerate(new_names) if n.lower() in ("other", "others")),
                len(new_names) - 1,
            )
            label_map[i] = oidx
        else:
            label_map[i] = 0

    return new_names, label_map


# ====================== Head Rebuild ======================

def rebuild_head(model, cfg, new_nc):
    """Replace the classification head for a new number of classes."""
    from models import MLPHead_CLS, MLPHead_TemporalMean
    hd = cfg["head"]["hidden_dim"]
    dr = cfg["head"]["dropout"]
    inf = cfg["head"]["in_features"]
    pool = cfg["head"].get("pool", "cls_token")
    if pool == "temporal_mean":
        model.head = MLPHead_TemporalMean(inf, new_nc, hd, dr)
    else:
        model.head = MLPHead_CLS(inf, new_nc, hd, dr)
    return model


# ====================== Training Loop ======================

def train(model, cfg, train_ds, val_ds, new_names,
          n_epochs, batch_sz, lr, output_dir,
          accum_steps=2, on_epoch_end=None):
    """
    Run the training loop with frame-wise validation.

    Args:
        model: torch.nn.Module (already on correct device)
        cfg: normalized config dict
        train_ds: SlidingWindowDataset
        val_ds: WindowPredictionDataset or None
        new_names: list of training class names
        n_epochs: int
        batch_sz: int
        lr: float
        output_dir: str, checkpoint save directory
        accum_steps: gradient accumulation steps
        on_epoch_end: optional callback(epoch, loss, metrics)

    Returns:
        list of epoch log dicts
    """
    device = next(model.parameters()).device
    new_nc = len(new_names)
    os.makedirs(output_dir, exist_ok=True)

    train_loader = DataLoader(
        train_ds, batch_sz, shuffle=True,
        num_workers=8, pin_memory=True,
        persistent_workers=True, prefetch_factor=2,
    )
    val_loader = DataLoader(
        val_ds, batch_sz, shuffle=False,
        num_workers=8, pin_memory=True,
        persistent_workers=True, prefetch_factor=2,
    ) if val_ds and len(val_ds) > 0 else None

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs)
    scaler = GradScaler()
    criterion = nn.CrossEntropyLoss()
    log = []

    for ep in range(n_epochs):
        model.train()
        running_loss = 0.0
        optimizer.zero_grad(set_to_none=True)
        nb = len(train_loader)

        for bi, (vids, tgts) in enumerate(train_loader):
            vids = vids.to(device)
            tgts = tgts.to(device)
            with autocast():
                loss = criterion(model(vids), tgts) / accum_steps
            scaler.scale(loss).backward()
            if (bi + 1) % accum_steps == 0 or (bi + 1) == nb:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            running_loss += loss.item() * accum_steps * vids.size(0)

        scheduler.step()
        ep_loss = running_loss / len(train_ds)

        f1m, mAP, f1p, app = 0.0, 0.0, [], []
        if val_loader:
            model.eval()
            window_probs = []
            with torch.no_grad():
                for vids, _ in val_loader:
                    vids = vids.to(device)
                    with autocast():
                        pr = torch.softmax(model(vids), dim=1).cpu().numpy()
                    window_probs.extend(pr)
            window_probs = np.array(window_probs)
            metrics = compute_framewise_metrics(window_probs, val_ds.frame_mappings, new_nc)
            f1m = metrics["f1"]
            mAP = metrics["mAP"]
            f1p = metrics["f1_per"]
            app = metrics["ap_per"]

        ckpt_path = os.path.join(output_dir, f"epoch_{ep+1}_f1_{f1m:.4f}_map_{mAP:.4f}.pth")
        torch.save(model.state_dict(), ckpt_path)

        epoch_log = {
            "epoch": ep + 1, "loss": ep_loss,
            "f1": f1m, "mAP": mAP,
            "f1_per": f1p, "ap_per": app,
            "path": ckpt_path,
        }
        log.append(epoch_log)
        print(f"Epoch {ep+1}/{n_epochs} | loss: {ep_loss:.4f} | F1: {f1m:.4f} | mAP: {mAP:.4f}")

        if on_epoch_end:
            on_epoch_end(ep + 1, ep_loss, {"f1": f1m, "mAP": mAP, "f1_per": f1p, "ap_per": app})

    with open(os.path.join(output_dir, "training_log.json"), "w") as f:
        json.dump(log, f, indent=2)

    return log
