"""
Config compatibility layer.

Handles multiple config formats produced by different training scripts,
normalizing them into a single unified format for inference.
"""

import copy
import glob
import json
import os

import torch


def normalize_config(raw):
    """
    Normalize a raw config dict (from any training script version) into a
    unified format.

    Returns:
        (cfg, error_message) — cfg is None if normalization fails.
    """
    if raw.get("model_type") == "pretrained_backbone":
        return None, "This is a pretrained backbone, not a fine-tuned model"

    cfg = {}

    # --- Format A: already has backbone dict with "name" ---
    if "backbone" in raw and isinstance(raw["backbone"], dict) and "name" in raw.get("backbone", {}):
        cfg = copy.deepcopy(raw)

    # --- Format B: uses "model_info" wrapper ---
    elif "model_info" in raw:
        mi = raw["model_info"]
        bb_name = mi.get("backbone", "unknown")
        bb_cfg = mi.get("backbone_config", {})
        hd = mi.get("head", {})
        inf = mi.get("input_format", {})
        cfg["backbone"] = {
            "name": bb_name,
            "hidden_size": hd.get("in_features", 768),
            "num_frames": bb_cfg.get("num_frames", inf.get("T", 8)),
            "input_size": bb_cfg.get("input_size", inf.get("H", 224)),
        }
        cfg["head"] = {
            "in_features": hd.get("in_features", 768),
            "hidden_dim": hd.get("hidden_dim", 512),
            "dropout": hd.get("dropout", 0.3),
            "pool": hd.get("pool", "temporal_mean"),
        }
        cfg["num_classes"] = raw.get("num_classes", len(raw.get("class_names", [])))
        cfg["class_names"] = raw.get("class_names", raw.get("SELECTED_BEHAVIORS", []))
        cfg["input_format"] = inf if inf else {
            "normalize": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}
        }

    # --- Format C: flat / unknown ---
    else:
        cfg = copy.deepcopy(raw)

    # --- Fill missing class_names ---
    if "class_names" not in cfg or not cfg["class_names"]:
        cfg["class_names"] = raw.get(
            "class_names", raw.get(
                "SELECTED_BEHAVIORS",
                [f"class_{i}" for i in range(raw.get("num_classes", 2))]
            )
        )

    if "num_classes" not in cfg:
        cfg["num_classes"] = len(cfg["class_names"])

    if "backbone" not in cfg or not isinstance(cfg["backbone"], dict):
        return None, "Cannot find backbone info in config"

    bb = cfg["backbone"]

    # Fill backbone defaults
    if "name" not in bb:
        bb["name"] = "CustomSwin3D"
    if "hidden_size" not in bb:
        bb["hidden_size"] = cfg.get("head", {}).get("in_features", 768)
    if "input_size" not in bb:
        bb["input_size"] = cfg.get("input_format", {}).get("H", 224)
    if "num_frames" not in bb:
        bb["num_frames"] = cfg.get("input_format", {}).get(
            "T", 8 if "timesformer" in bb.get("name", "").lower() else 16
        )
    if "pretrained" not in bb:
        name = bb["name"]
        if name == "TimesformerModel" or "timesformer" in name.lower():
            bb["pretrained"] = "facebook/timesformer-base-finetuned-k400"
        elif name == "CustomSwin3D" or "swin" in name.lower():
            bb["pretrained"] = "Swin3D_T_Weights.DEFAULT"

    # Fill head defaults
    if "head" not in cfg:
        cfg["head"] = {}
    hd = cfg["head"]
    hd.setdefault("in_features", bb.get("hidden_size", 768))
    hd.setdefault("hidden_dim", 512)
    hd.setdefault("dropout", 0.3)

    # Fill input_format defaults
    if "input_format" not in cfg:
        cfg["input_format"] = {}
    if "normalize" not in cfg["input_format"]:
        cfg["input_format"]["normalize"] = {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        }

    if not cfg["class_names"]:
        return None, "No class names found in config"
    if cfg["num_classes"] != len(cfg["class_names"]):
        cfg["num_classes"] = len(cfg["class_names"])

    return cfg, None


def find_config_for_pth(pth_path):
    """
    Try to locate and load a config for a given .pth file.

    Search order:
      1. <stem>_config.json in the same directory
      2. config.json in the same directory
      3. Any *_config.json in the same directory
      4. Auto-detect from checkpoint keys (best-effort, class names unknown)

    Returns:
        (cfg, source_description, error_message)
        cfg is None and error_message is set on failure.
    """
    pth_dir = os.path.dirname(pth_path)
    pth_stem = os.path.basename(pth_path).rsplit(".", 1)[0]

    # 1. Named config
    named_config = os.path.join(pth_dir, pth_stem + "_config.json")
    if os.path.exists(named_config):
        try:
            with open(named_config) as f:
                raw = json.load(f)
            cfg, err = normalize_config(raw)
            if cfg:
                return cfg, f"✅ {os.path.basename(named_config)}", None
        except Exception:
            pass

    # 2. config.json in same dir
    dir_config = os.path.join(pth_dir, "config.json")
    if os.path.exists(dir_config):
        try:
            with open(dir_config) as f:
                raw = json.load(f)
            cfg, err = normalize_config(raw)
            if cfg:
                return cfg, "✅ config.json (same folder)", None
        except Exception:
            pass

    # 3. Any *_config.json in same dir
    config_files = sorted(glob.glob(os.path.join(pth_dir, "*_config.json")))
    for cf in config_files:
        try:
            with open(cf) as f:
                raw = json.load(f)
            cfg, err = normalize_config(raw)
            if cfg:
                return cfg, f"⚠️ {os.path.basename(cf)} (nearest match)", None
        except Exception:
            continue

    # 4. Auto-detect from checkpoint keys
    try:
        sd = torch.load(pth_path, map_location="cpu", weights_only=True)
        has_timesformer = any(
            "backbone.encoder" in k or "backbone.embeddings" in k for k in sd.keys()
        )
        has_swin = any(
            "model.features" in k or "model.patch_embed" in k for k in sd.keys()
        )
        if has_timesformer:
            bb_name, nf = "TimesformerModel", 8
        elif has_swin:
            bb_name, nf = "CustomSwin3D", 8
        else:
            del sd
            return None, None, "Cannot determine backbone type"

        nc = 2
        for key in ["head.fc2.weight", "head.fc2.bias"]:
            if key in sd:
                nc = sd[key].shape[0]
                break
        del sd

        cfg = {
            "backbone": {"name": bb_name, "hidden_size": 768, "num_frames": nf, "input_size": 224},
            "head": {"in_features": 768, "hidden_dim": 512, "dropout": 0.3},
            "num_classes": nc,
            "class_names": [f"class_{i}" for i in range(nc)],
            "input_format": {
                "normalize": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}
            },
        }
        cfg, _ = normalize_config(cfg)
        return cfg, f"⚠️ Auto-detected: {bb_name}, {nc} classes (names unknown)", None

    except Exception as e:
        return None, None, f"Cannot determine config: {e}"
