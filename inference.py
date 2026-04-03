"""
Core inference utilities for animal behavior classification.

Supports sliding-window inference over videos using TimeSformer or Swin3D backbones.
"""

import numpy as np
import torch
from collections import Counter
from decord import VideoReader, cpu
from PIL import Image
from torchvision.transforms import ToTensor


# ====================== Preprocessing ======================

def uniform_sample(frames, t):
    """Sample exactly t frames uniformly from a list (with repetition if needed)."""
    n = len(frames)
    if n == t:
        return frames
    if n < t:
        return frames + [frames[-1]] * (t - n)
    return [frames[i] for i in np.linspace(0, n - 1, t, dtype=int)]


def preprocess(frames, cfg):
    """
    Convert a list of PIL Images to a normalized tensor.

    Args:
        frames: list of PIL.Image
        cfg: normalized config dict

    Returns:
        Tensor of shape (C, T, H, W)
    """
    sz = cfg["backbone"]["input_size"]
    nf = cfg["backbone"]["num_frames"]
    mean = cfg["input_format"]["normalize"]["mean"]
    std = cfg["input_format"]["normalize"]["std"]

    resized = [f.resize((sz, sz), Image.BILINEAR) for f in frames]
    if len(resized) != nf:
        resized = uniform_sample(resized, nf)

    v = torch.stack([ToTensor()(f) for f in resized], dim=0)
    v = (v - torch.tensor(mean).view(1, -1, 1, 1)) / torch.tensor(std).view(1, -1, 1, 1)
    return v.permute(1, 0, 2, 3)  # (C, T, H, W)


# ====================== Behavior Remapping ======================

def get_others_idx(cfg):
    """Return index of the 'others/other' class, or -1 if not present."""
    for i, nm in enumerate(cfg["class_names"]):
        if nm.lower() in ("others", "other"):
            return i
    return -1


def remap_with_disabled(pred_label, probs, cfg, disabled_classes):
    """
    Redirect probability mass from disabled classes to the 'others' class.

    Args:
        pred_label: int, predicted class index
        probs: np.ndarray of shape (num_classes,)
        cfg: normalized config dict
        disabled_classes: set of class indices to disable

    Returns:
        (new_pred_label, new_probs)
    """
    if not disabled_classes:
        return pred_label, probs

    others_idx = get_others_idx(cfg)
    if others_idx < 0:
        return pred_label, probs  # no 'others' class to redirect into

    new_probs = probs.copy()
    for dc in disabled_classes:
        if dc < len(new_probs):
            new_probs[others_idx] += new_probs[dc]
            new_probs[dc] = 0.0

    new_label = int(np.argmax(new_probs))
    return new_label, new_probs


# ====================== Sliding-Window Inference ======================

def infer_video_gen(vdir, vf, model, cfg, disabled_classes=None, yield_every=5):
    """
    Run sliding-window inference on a single video file.

    Yields progress tuples (windows_done, total_windows) during inference,
    then yields the final result dict.

    Args:
        vdir: str, path to video directory
        vf: str, video filename
        model: torch.nn.Module
        cfg: normalized config dict
        disabled_classes: set of class indices to merge into 'others'
        yield_every: int, yield progress every N windows

    Yields:
        (int, int) — progress
        dict — final result with keys:
            frame_labels, frame_confidences, total_frames, fps, video_path
    """
    if disabled_classes is None:
        disabled_classes = set()

    device = next(model.parameters()).device
    ws = cfg["backbone"]["num_frames"]   # window size
    st = max(1, ws // 4)                 # stride
    nc = cfg["num_classes"]

    import os
    vp = os.path.join(vdir, vf)
    vr = VideoReader(vp, ctx=cpu(0))
    T = len(vr)
    fps = vr.get_avg_fps()

    votes = [[] for _ in range(T)]
    probs = [[] for _ in range(T)]
    wins = list(range(0, T - ws + 1, st))

    if not wins:
        # Video shorter than one window — use all frames
        fr = [Image.fromarray(f) for f in vr.get_batch(list(range(T))).asnumpy()]
        while len(fr) < ws:
            fr.append(fr[-1])
        with torch.no_grad():
            p = torch.softmax(
                model(preprocess(fr, cfg).unsqueeze(0).to(device)), dim=1
            )[0].cpu().numpy()
        pred, p = remap_with_disabled(int(np.argmax(p)), p, cfg, disabled_classes)
        for i in range(T):
            votes[i].append(pred)
            probs[i].append(p)
        yield (1, 1)
    else:
        for wi, s in enumerate(wins):
            idx = list(range(s, s + ws))
            fr = [Image.fromarray(f) for f in vr.get_batch(idx).asnumpy()]
            with torch.no_grad():
                p = torch.softmax(
                    model(preprocess(fr, cfg).unsqueeze(0).to(device)), dim=1
                )[0].cpu().numpy()
            pred, p = remap_with_disabled(int(np.argmax(p)), p, cfg, disabled_classes)
            for i in idx:
                votes[i].append(pred)
                probs[i].append(p)
            if (wi + 1) % yield_every == 0 or wi == len(wins) - 1:
                yield (wi + 1, len(wins))

    # Aggregate per-frame votes
    labels, confs = [], []
    for i in range(T):
        if votes[i]:
            labels.append(Counter(votes[i]).most_common(1)[0][0])
            confs.append(np.mean(probs[i], axis=0))
        else:
            labels.append(nc - 1)
            confs.append(np.zeros(nc))

    yield {
        "frame_labels": labels,
        "frame_confidences": confs,
        "total_frames": T,
        "fps": fps,
        "video_path": vp,
    }
