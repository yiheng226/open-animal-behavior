"""
Gradio GUI for animal behavior model fine-tuning.

Usage:
    python gui_training.py
"""

import copy, json, os, random
import numpy as np
import torch
import gradio as gr
import pandas as pd
from collections import Counter
from decord import VideoReader, cpu
from difflib import SequenceMatcher
from huggingface_hub import hf_hub_download, list_repo_files
from PIL import Image
from sklearn.model_selection import train_test_split

from models import build_model_from_config
from config_utils import normalize_config
from training_utils import (
    SlidingWindowDataset, WindowPredictionDataset,
    compute_framewise_metrics, compute_label_map, rebuild_head,
    fuzzy_match, random_blur, temporal_dropout, train,
)

# ==================== 👇 修改這裡 👇 ====================
HF_REPO_ID         = "yiheng266/animal-social-models"
DEFAULT_VIDEO_DIR  = "/content/drive/My Drive/videos/train/"
DEFAULT_LABEL_DIR  = "/content/drive/My Drive/labels/train/"
DEFAULT_OUTPUT_DIR = "/content/drive/My Drive/trained_models/"
MAX_LABELS         = 15
# ==================== 👆 修改以上即可 👆 ====================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ====================== State ======================

S = {"model": None, "cfg": None, "scan_data": None, "label_names": [],
     "cur_vf": None, "cur_vr": None,
     "_cursor_data": json.dumps({"T": 0, "names": [], "labels": []}),
     "train_log": [], "split_indices": {"train": [], "val": []}}

CLR_PAL = ["#378ADD","#D85A30","#E24B4A","#7F77DD","#1D9E75","#BA7517",
           "#534AB7","#993C1D","#639922","#D4537E","#185FA5","#854F0B","#A32D2D"]
U = gr.update()

def get_clr(i, name):
    if name.lower() in ("other", "others"): return "#FFFFFF", "rgba(180,180,180,0.9)"
    c = CLR_PAL[i % len(CLR_PAL)]; r, g, b = int(c[1:3],16), int(c[3:5],16), int(c[5:7],16)
    return c, f"rgba({r},{g},{b},0.9)"

# ====================== Model Management ======================

def list_models(repo):
    try:
        files = list_repo_files(repo)
        pths = [f for f in files if f.endswith("/model.pth") or f == "model.pth"]
        if not pths: pths = [f for f in files if f.endswith(".pth")]
        if not pths: return gr.update(choices=[], value=None), "❌ No models found"
        names = [os.path.dirname(p) if "/" in p else p for p in pths]
        return gr.update(choices=names, value=names[0]), f"✅ {len(names)} model(s) found"
    except Exception as e:
        return gr.update(choices=[], value=None), f"❌ {e}"

def load_pretrained(repo, mname):
    if not mname or not repo: return "❌ Specify repo & model"
    try:
        cf = f"{mname}/config.json" if not mname.endswith(".pth") else "config.json"
        pf = f"{mname}/model.pth"  if not mname.endswith(".pth") else mname
        with open(hf_hub_download(repo_id=repo, filename=cf)) as f:
            raw = json.load(f)
        cfg, err = normalize_config(raw)
        if err: return f"❌ {err}"
        model = build_model_from_config(cfg)
        model.load_state_dict(torch.load(hf_hub_download(repo_id=repo, filename=pf), map_location=device, weights_only=True))
        model.to(device)
        S.update({"model": model, "cfg": cfg, "train_log": []})
        return (f"✅ Loaded: {mname}\n"
                f"  Backbone: {cfg['backbone']['name']}\n"
                f"  Classes: {cfg['class_names']}\n"
                f"  Device: {device}")
    except Exception as e:
        import traceback; traceback.print_exc(); return f"❌ {e}"

# ====================== Train/Val Split ======================

def compute_split(val_pct, seed=1337):
    if not S["scan_data"]: S["split_indices"] = {"train": [], "val": []}; return
    data = S["scan_data"]; n = len(data); val_ratio = val_pct / 100.0
    if val_ratio > 0 and n >= 4:
        tidx, vidx = train_test_split(list(range(n)), test_size=val_ratio, random_state=int(seed))
    elif val_ratio > 0 and n >= 2:
        vidx = [n - 1]; tidx = list(range(n - 1))
    else:
        tidx = list(range(n)); vidx = []
    S["split_indices"] = {"train": tidx, "val": vidx}

def build_video_list_html(active_vf=None):
    if not S["scan_data"]: return "<p style='color:#aaa;font-size:12px;'>Load data first</p>"
    data = S["scan_data"]
    tidx_set = set(S["split_indices"].get("train", []))
    vidx_set = set(S["split_indices"].get("val", []))
    html = "<div style='max-height:200px;overflow-y:auto;border:1px solid #e0e0e0;border-radius:8px;padding:4px;'>"
    for i, d in enumerate(data):
        vf = d["vf"]; T = d["T"]; fps = d["fps"]; dur = T / fps if fps > 0 else 0
        is_active = (vf == active_vf)
        bg = "background:rgba(220,38,38,0.12);border-left:3px solid #dc2626;" if is_active else "background:transparent;border-left:3px solid transparent;"
        if i in vidx_set:
            role_tag = "<span style='font-size:10px;padding:1px 5px;border-radius:3px;background:#FEF3C7;color:#92400E;font-weight:600;margin-left:6px;'>VAL</span>"
        elif i in tidx_set:
            role_tag = "<span style='font-size:10px;padding:1px 5px;border-radius:3px;background:#D1FAE5;color:#065F46;font-weight:600;margin-left:6px;'>TRAIN</span>"
        else:
            role_tag = ""
        nc = "#dc2626" if is_active else "#333"; nw = "700" if is_active else "500"
        html += f"<div style='{bg}border-radius:4px;padding:4px 8px;margin-bottom:1px;'><div style='display:flex;justify-content:space-between;align-items:center;'><span style='font-size:12px;color:{nc};font-weight:{nw};'>{vf}{role_tag}</span><span style='font-size:10px;color:#888;white-space:nowrap;'>{T} fr · {dur:.1f}s</span></div></div>"
    html += "</div>"
    nt = len(tidx_set); nv = len(vidx_set)
    html += f"<div style='display:flex;gap:14px;margin-top:4px;font-size:11px;color:#888;'><span>Train: {nt}</span><span>Val: {nv}</span><span>Total: {len(data)}</span></div>"
    return html

# ====================== Mapped Timeline ======================

def build_mapped_timeline(vf, mapped_names, label_map):
    if not S["scan_data"] or not vf: return "", S["_cursor_data"]
    d = next((x for x in S["scan_data"] if x["vf"] == vf), None)
    if not d: return "", S["_cursor_data"]
    T = d["T"]; fps = d["fps"]; raw_labels = d["labels"]
    mapped_labels = [label_map.get(l, 0) if label_map.get(l) is not None else -1 for l in raw_labels]
    names = mapped_names
    segs = []; cur, cnt = mapped_labels[0], 1
    for i in range(1, T):
        if mapped_labels[i] == cur: cnt += 1
        else: segs.append((cur, cnt)); cur, cnt = mapped_labels[i], 1
    segs.append((cur, cnt))
    bar = ""
    for li, c in segs:
        if li == -1: nm = "excluded"; clr = "#D0D0D0"; bdr = ""
        else:
            nm = names[li] if li < len(names) else "?"
            clr, _ = get_clr(li, nm)
            bdr = "border-top:1px solid #ccc;border-bottom:1px solid #ccc;" if clr == "#FFFFFF" else ""
        pct = (c / T) * 100
        bar += f"<div style='width:{pct:.3f}%;height:100%;background:{clr};{bdr}display:inline-block;box-sizing:border-box;opacity:{0.35 if li==-1 else 1};' title='{nm}'></div>"
    leg = "".join(
        f"<span style='display:inline-flex;align-items:center;gap:3px;margin-right:10px;font-size:11px;color:#888;'>"
        f"<span style='display:inline-block;width:8px;height:8px;border-radius:2px;background:{get_clr(i,nm)[0]};{'border:1px solid #ccc;' if get_clr(i,nm)[0]=='#FFFFFF' else ''}'></span>{nm}</span>"
        for i, nm in enumerate(names)
    )
    ml0 = mapped_labels[0]; nm0 = names[ml0] if ml0 >= 0 and ml0 < len(names) else "excluded"
    tl = f"""<div style='width:100%;padding:4px 0;'>
      <div style='position:relative;display:flex;height:16px;border-radius:4px;overflow:hidden;border:1px solid #ccc;'>
        {bar}
        <div id='tl-cursor' style='position:absolute;top:-2px;bottom:-2px;width:2px;background:#000;left:0%;pointer-events:none;'></div>
      </div>
      <div style='display:flex;justify-content:space-between;align-items:center;margin-top:3px;'>
        <div>{leg}</div>
        <span id='tl-frame-label' style='font-size:11px;font-weight:500;color:#888;'>F:0 {nm0}</span>
      </div></div>"""
    cursor_data = json.dumps({"T": T, "names": names, "labels": mapped_labels})
    S["_cursor_data"] = cursor_data
    return tl, cursor_data

def build_mapping_summary_html(mode, dd_values, data_labels, pretrained_names):
    new_names, _ = compute_label_map(mode, dd_values, data_labels, pretrained_names)
    return (f"<div style='padding:6px 10px;background:#f7f7f7;border-radius:8px;font-size:11px;color:#888;line-height:1.6;'>"
            f"<span style='font-weight:500;'>Training classes ({len(new_names)}):</span> {', '.join(new_names)}</div>")

# ====================== Mapping Choices ======================

def build_mapping_choices_pt(idx, data_labels, pretrained_names):
    has_other = any(n.lower() in ("other","others") for n in data_labels)
    choices = list(pretrained_names)
    if not has_other: choices.append("→ Other")
    choices.append("→ Exclude")
    default = fuzzy_match(data_labels[idx], pretrained_names)
    if default is None:
        default = "others" if "others" in pretrained_names else ("→ Other" if not has_other else choices[0])
    return choices, default

def build_mapping_choices_new(idx, data_labels, all_mappings):
    has_other = any(n.lower() in ("other","others") for n in data_labels)
    consumed = {i for i_str, val in all_mappings.items()
                if (i := int(i_str)) != idx and val not in (None, "", "keep", "→ Other", "→ Exclude")}
    choices = [f"{data_labels[idx]} (keep)"]
    for j, nm in enumerate(data_labels):
        if j != idx and j not in consumed: choices.append(f"→ merge into {nm}")
    if not has_other: choices.append("→ Other")
    choices.append("→ Exclude")
    return choices

# ====================== Label Distribution ======================

def build_label_dist_html():
    if not S["scan_data"] or not S["label_names"]: return "<p style='color:#aaa;'>Load data to see labels</p>"
    names = S["label_names"]; total = sum(d["T"] for d in S["scan_data"])
    gcounts = Counter()
    for d in S["scan_data"]:
        for k, v in d["counts"].items(): gcounts[k] += v
    h = "<div style='padding:4px 0;'><p style='font-size:13px;font-weight:500;margin:0 0 6px;'>Label distribution</p>"
    for i, nm in enumerate(names):
        c = gcounts.get(i, 0); pct = 100 * c / max(total, 1); clr, _ = get_clr(i, nm)
        bar_clr = "#ddd" if clr == "#FFFFFF" else clr
        h += f"<div style='margin-bottom:5px;'><div style='display:flex;align-items:center;gap:6px;margin-bottom:1px;'><span style='display:inline-block;width:8px;height:8px;border-radius:2px;background:{bar_clr};flex-shrink:0;'></span><span style='font-size:12px;font-weight:500;flex:1;'>{nm}</span><span style='font-size:11px;color:#888;flex-shrink:0;'>{c:,} fr · {pct:.1f}%</span></div><div style='height:5px;background:#f0f0f0;border-radius:3px;overflow:hidden;margin-left:14px;'><div style='width:{max(pct,0.3):.1f}%;height:100%;background:{bar_clr};border-radius:3px;'></div></div></div>"
    h += "</div>"
    return h

# ====================== Data Scanning ======================

def do_scan_and_preview(vdir, ldir, val_pct, val_seed, head_mode, *dd_vals):
    N = MAX_LABELS
    empty = lambda msg: (msg, "", "*Load data first*",
                         *[gr.update(visible=False, choices=[], value=None) for _ in range(N)],
                         gr.update(choices=[], value=None), None, "", "", gr.update(maximum=0, value=0), S["_cursor_data"], "", "")
    if not vdir or not os.path.isdir(vdir): return empty("❌ Video dir not found")
    if not ldir or not os.path.isdir(ldir): return empty("❌ Label dir not found")
    vfiles = sorted([f for f in os.listdir(vdir) if f.lower().endswith((".mp4", ".avi"))])
    if not vfiles: return empty("❌ No videos found")
    matched = []; all_label_names = None
    for vf in vfiles:
        base = os.path.splitext(vf)[0]; lp = None
        for c in [base + ".csv", base + "_one_hot.csv"]:
            fp = os.path.join(ldir, c)
            if os.path.exists(fp): lp = fp; break
        if lp is None: continue
        vp = os.path.join(vdir, vf)
        try:
            vr = VideoReader(vp, ctx=cpu(0)); T = len(vr); fps = vr.get_avg_fps()
            df = pd.read_csv(lp); nc = df.shape[1]
            if all_label_names is None: all_label_names = list(df.columns[:nc])
            labels = np.argmax(df.values[:, :nc], axis=1)
            matched.append({"vp": vp, "lp": lp, "vf": vf, "T": T, "fps": fps,
                             "counts": Counter(labels.tolist()), "labels": labels.tolist()})
        except: continue
    if not matched: return empty("❌ No matched pairs")
    if all_label_names is None: all_label_names = [f"class_{i}" for i in range(2)]
    S["scan_data"] = matched; S["label_names"] = all_label_names
    compute_split(val_pct, val_seed)
    pretrained_names = S["cfg"]["class_names"] if S["cfg"] else []
    dd_updates = []
    for i in range(N):
        if i < len(all_label_names):
            if head_mode == "Pretrain head" and pretrained_names:
                choices, default = build_mapping_choices_pt(i, all_label_names, pretrained_names)
            else:
                choices = build_mapping_choices_new(i, all_label_names, {}); default = choices[0]
            dd_updates.append(gr.update(visible=True, choices=choices, value=default, label=all_label_names[i]))
        else:
            dd_updates.append(gr.update(visible=False, choices=[], value=None))
    vnames = [d["vf"] for d in matched]; vf = matched[0]["vf"]
    dd_values = [u["value"] for u in dd_updates[:len(all_label_names)]]
    new_names, label_map = compute_label_map(head_mode, dd_values, all_label_names, pretrained_names)
    tl, cdata = build_mapped_timeline(vf, new_names, label_map)
    summary = build_mapping_summary_html(head_mode, dd_values, all_label_names, pretrained_names)
    img = _get_frame(vf, 0)
    d0 = next(x for x in matched if x["vf"] == vf); T = d0["T"]; fps = d0["fps"]
    ml = label_map.get(d0["labels"][0])
    nm0 = new_names[ml] if ml is not None and ml < len(new_names) else "excluded"
    _, bg = get_clr(ml, nm0) if ml is not None else ("#888", "rgba(180,180,180,0.6)")
    info = f"<div style='display:flex;justify-content:space-between;align-items:center;'><span style='padding:3px 10px;border-radius:6px;background:{bg};color:white;font-size:12px;font-weight:500;'>{nm0}</span><span style='font-size:12px;color:#888;'>F: 0 / {T} | 0.00s / {T/fps:.2f}s</span></div>"
    return (f"✅ {len(matched)} matched (of {len(vfiles)} videos)",
            build_label_dist_html(), f"**{vf}** — 1 / {len(matched)} videos",
            *dd_updates, gr.update(choices=vnames, value=vf),
            img, info, tl, gr.update(maximum=max(T-1, 0), value=0), cdata,
            build_video_list_html(active_vf=vf), summary)

# ====================== Mapping Change ======================

def on_mapping_change(head_mode, *dd_vals):
    data_labels = S["label_names"]
    pretrained_names = S["cfg"]["class_names"] if S["cfg"] else []
    N = MAX_LABELS; n = len(data_labels)
    if n == 0: return (*[gr.update() for _ in range(N)], "", S["_cursor_data"], "")
    cur_vals = list(dd_vals[:N])
    dd_updates = []
    for i in range(N):
        if i < n:
            if head_mode == "Pretrain head":
                choices, default = build_mapping_choices_pt(i, data_labels, pretrained_names)
                current = cur_vals[i]
                dd_updates.append(gr.update(choices=choices, value=current if current in choices else default))
            else:
                mappings = {str(j): cur_vals[j] for j in range(n)}
                choices = build_mapping_choices_new(i, data_labels, mappings)
                current = cur_vals[i]
                dd_updates.append(gr.update(choices=choices, value=current if current in choices else choices[0]))
        else:
            dd_updates.append(gr.update())
    final_vals = [dd_updates[i].get("value", cur_vals[i]) if isinstance(dd_updates[i], dict) and "value" in dd_updates[i] else cur_vals[i] for i in range(n)]
    new_names, label_map = compute_label_map(head_mode, final_vals, data_labels, pretrained_names)
    vf = S["cur_vf"]
    tl, cdata = build_mapped_timeline(vf, new_names, label_map) if vf else ("", S["_cursor_data"])
    return (*dd_updates, tl, cdata, build_mapping_summary_html(head_mode, final_vals, data_labels, pretrained_names))

# ====================== Video Preview ======================

def _get_frame(vf, fi):
    if not S["scan_data"]: return None
    d = next((x for x in S["scan_data"] if x["vf"] == vf), None)
    if not d: return None
    try:
        if S["cur_vf"] != vf or S["cur_vr"] is None:
            S["cur_vr"] = VideoReader(d["vp"], ctx=cpu(0)); S["cur_vf"] = vf
        T = len(S["cur_vr"]); fi = max(0, min(int(fi), T - 1))
        return S["cur_vr"][fi].asnumpy()
    except: S["cur_vr"] = None; S["cur_vf"] = None; return None

def _preview_video_mapped(vf, head_mode, dd_vals):
    if not S["scan_data"] or not vf: return None, "", "", U, S["_cursor_data"]
    d = next((x for x in S["scan_data"] if x["vf"] == vf), None)
    if not d: return None, "", "", U, S["_cursor_data"]
    data_labels = S["label_names"]; pretrained_names = S["cfg"]["class_names"] if S["cfg"] else []
    new_names, label_map = compute_label_map(head_mode, list(dd_vals[:len(data_labels)]), data_labels, pretrained_names)
    T = d["T"]; fps = d["fps"]
    tl, cdata = build_mapped_timeline(vf, new_names, label_map)
    ml = label_map.get(d["labels"][0])
    nm0 = new_names[ml] if ml is not None and ml < len(new_names) else "excluded"
    _, bg = get_clr(ml, nm0) if ml is not None else ("#888", "rgba(180,180,180,0.6)")
    info = f"<div style='display:flex;justify-content:space-between;align-items:center;'><span style='padding:3px 10px;border-radius:6px;background:{bg};color:white;font-size:12px;font-weight:500;'>{nm0}</span><span style='font-size:12px;color:#888;'>F: 0 / {T} | 0.00s / {T/fps:.2f}s</span></div>"
    return _get_frame(vf, 0), info, tl, gr.update(maximum=max(T-1, 0), value=0), cdata

def on_scrub(fi, head_mode, *dd_vals):
    vf = S["cur_vf"]
    if not vf or not S["scan_data"]: return None, ""
    d = next((x for x in S["scan_data"] if x["vf"] == vf), None)
    if not d: return None, ""
    data_labels = S["label_names"]; pretrained_names = S["cfg"]["class_names"] if S["cfg"] else []
    new_names, label_map = compute_label_map(head_mode, list(dd_vals[:len(data_labels)]), data_labels, pretrained_names)
    T = d["T"]; fps = d["fps"]; fi = max(0, min(int(fi), T - 1))
    ml = label_map.get(d["labels"][fi])
    nm = new_names[ml] if ml is not None and ml < len(new_names) else "excluded"
    _, bg = get_clr(ml, nm) if ml is not None else ("#888", "rgba(180,180,180,0.6)")
    info = f"<div style='display:flex;justify-content:space-between;align-items:center;'><span style='padding:3px 10px;border-radius:6px;background:{bg};color:white;font-size:12px;font-weight:500;'>{nm}</span><span style='font-size:12px;color:#888;'>F: {fi} / {T} | {fi/fps:.2f}s / {T/fps:.2f}s</span></div>"
    return _get_frame(vf, fi), info

def do_nav(direction, head_mode, *dd_vals):
    if not S["scan_data"]: return None, "", "", U, S["_cursor_data"], "*No data*", ""
    vnames = [d["vf"] for d in S["scan_data"]]; cur = S["cur_vf"]
    idx = vnames.index(cur) if cur in vnames else 0
    idx = max(0, idx - 1) if direction == "prev" else min(len(vnames) - 1, idx + 1)
    vf = vnames[idx]
    img, info, tl, scrub, cdata = _preview_video_mapped(vf, head_mode, dd_vals)
    return img, info, tl, scrub, cdata, f"**{vf}** — {idx+1} / {len(vnames)} videos", build_video_list_html(active_vf=vf)

def on_vid_change(vf, head_mode, *dd_vals):
    img, info, tl, scrub, cdata = _preview_video_mapped(vf, head_mode, dd_vals)
    idx = next((i for i, d in enumerate(S["scan_data"]) if d["vf"] == vf), 0) if S["scan_data"] else 0
    return img, info, tl, scrub, cdata, build_video_list_html(active_vf=vf), f"**{vf}** — {idx+1} / {len(S['scan_data'])} videos"

def on_val_ratio_change(val_pct, val_seed):
    compute_split(val_pct, val_seed)
    return build_video_list_html(active_vf=S["cur_vf"])

# ====================== Training Progress HTML ======================

def html_progress(ep_done, ep_total, win_done, win_total, phase="training"):
    if ep_total == 0: return ""
    ep_pct = (ep_done / ep_total) * 100; wp = (win_done / max(win_total, 1)) * 100
    ec = "#1D9E75" if ep_done == ep_total else "#D85A30"
    st = "✅ Complete" if ep_done == ep_total else "Training..."
    return f"""<div style='background:#fff;border:1px solid #e0e0e0;border-radius:8px;padding:10px 14px;'>
      <div style='display:flex;justify-content:space-between;margin-bottom:4px;'>
        <span style='font-size:13px;font-weight:500;'>Epoch — {st}</span>
        <span style='font-size:12px;color:#888;'>{ep_done}/{ep_total} epochs</span></div>
      <div style='height:8px;background:#eee;border-radius:4px;overflow:hidden;margin-bottom:10px;'>
        <div style='width:{ep_pct:.1f}%;height:100%;background:{ec};border-radius:4px;'></div></div>
      <div style='display:flex;justify-content:space-between;margin-bottom:4px;'>
        <span style='font-size:12px;font-weight:500;'>Epoch {min(ep_done+1,ep_total)} — {phase}</span>
        <span style='font-size:12px;color:#888;'>{win_done}/{win_total} windows</span></div>
      <div style='height:6px;background:#eee;border-radius:3px;overflow:hidden;'>
        <div style='width:{wp:.1f}%;height:100%;background:#1D9E75;border-radius:3px;'></div></div>
    </div>"""

def html_val_card(epoch, loss, f1, mAP, f1_per, ap_per, names, is_best=False):
    badge = "<span style='font-size:10px;padding:2px 6px;background:#DBEAFE;color:#1E40AF;border-radius:4px;margin-left:6px;'>best</span>" if is_best else ""
    fc = "color:#1E40AF;" if is_best else ""
    brd = "border:2px solid #93C5FD;" if is_best else "border:1px solid #e0e0e0;"
    rows = "".join(f"<div style='display:flex;justify-content:space-between;'><span>{nm}</span><span>F1: {f1_per[i] if i<len(f1_per) else 0:.3f} · AP: {ap_per[i] if i<len(ap_per) else 0:.3f}</span></div>" for i, nm in enumerate(names))
    return f"<div style='background:#fff;{brd}border-radius:8px;padding:14px;margin-bottom:12px;'><div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;'><div><span style='font-size:13px;font-weight:500;'>Epoch {epoch}</span>{badge}</div><span style='font-size:11px;color:#888;'>loss: {loss:.4f}</span></div><div style='display:flex;gap:8px;margin-bottom:8px;'><div style='flex:1;background:#f7f7f7;border-radius:6px;padding:6px;text-align:center;'><div style='font-size:11px;color:#888;'>F1-macro</div><div style='font-size:16px;font-weight:500;{fc}'>{f1:.4f}</div></div><div style='flex:1;background:#f7f7f7;border-radius:6px;padding:6px;text-align:center;'><div style='font-size:11px;color:#888;'>mAP</div><div style='font-size:16px;font-weight:500;{fc}'>{mAP:.4f}</div></div></div><div style='font-size:11px;color:#888;line-height:1.6;'>{rows}</div></div>"

def build_val_html(log, names):
    if not log: return "<p style='color:#aaa;'>Training not started</p>"
    best = max(range(len(log)), key=lambda i: log[i]["f1"])
    return "".join(html_val_card(e["epoch"], e["loss"], e["f1"], e["mAP"], e["f1_per"], e["ap_per"], names, i == best) for i, e in enumerate(log))

# ====================== Training ======================

def run_training(repo, mname, vdir, ldir, odir, head_mode,
                 n_epochs, batch_sz, lr_str, val_pct, ws, stride_val, val_seed, train_seed, *dd_vals):
    try:
        lr = float(lr_str); n_epochs = int(n_epochs); batch_sz = int(batch_sz)
        ws = int(ws); stride_val = int(stride_val); val_seed = int(val_seed); train_seed = int(train_seed)
    except Exception as e:
        yield f"❌ Invalid params: {e}", U; return
    if not S["model"] or not S["cfg"]: yield "❌ Load model first", U; return
    if not S["scan_data"]: yield "❌ Load data first", U; return

    cfg = S["cfg"]; data_labels = S["label_names"]
    pretrained_names = cfg["class_names"]; os.makedirs(odir, exist_ok=True)
    vals = list(dd_vals[:len(data_labels)])
    new_names, label_map = compute_label_map(head_mode, vals, data_labels, pretrained_names)
    new_nc = len(new_names)

    model = copy.deepcopy(S["model"])
    if head_mode == "New head":
        model = rebuild_head(model, cfg, new_nc)
    model = model.to(device)

    data = S["scan_data"]; vps = [d["vp"] for d in data]; lps = [d["lp"] for d in data]
    tidx = S["split_indices"]["train"]; vidx = S["split_indices"]["val"]

    yield html_progress(0, n_epochs, 0, 0, "building dataset..."), "<p style='color:#aaa;'>Building...</p>"

    aug_rng = random.Random(train_seed)
    def aug(fr): return temporal_dropout(random_blur(fr, rng=aug_rng), rng=aug_rng)

    train_ds = SlidingWindowDataset([vps[i] for i in tidx], [lps[i] for i in tidx], ws, stride_val, cfg, new_nc, label_map, augment=aug)
    val_ds = SlidingWindowDataset([vps[i] for i in vidx], [lps[i] for i in vidx], ws, stride_val, cfg, new_nc, label_map) if vidx else None

    if len(train_ds) == 0: yield "❌ No training windows created.", ""; return

    from torch.utils.data import DataLoader
    from torch.cuda.amp import GradScaler, autocast
    import torch.nn as nn
    import torch.optim as optim
    from torch.optim.lr_scheduler import CosineAnnealingLR

    train_loader = DataLoader(train_ds, batch_sz, shuffle=True, num_workers=8, pin_memory=True, persistent_workers=True, prefetch_factor=2)
    val_loader = DataLoader(val_ds, batch_sz, shuffle=False, num_workers=8, pin_memory=True, persistent_workers=True, prefetch_factor=2) if val_ds and len(val_ds) > 0 else None

    total_win = len(train_ds)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs)
    scaler = GradScaler(); criterion = nn.CrossEntropyLoss(); accum = 2
    S["train_log"] = []

    for ep in range(n_epochs):
        model.train(); rl = 0.0; optimizer.zero_grad(set_to_none=True); nb = len(train_loader)
        for bi, (vids, tgts) in enumerate(train_loader):
            vids = vids.to(device); tgts = tgts.to(device)
            with autocast(): loss = criterion(model(vids), tgts) / accum
            scaler.scale(loss).backward()
            if (bi + 1) % accum == 0 or (bi + 1) == nb:
                scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
            rl += loss.item() * accum * vids.size(0)
            if (bi + 1) % 5 == 0 or bi == nb - 1:
                yield html_progress(ep, n_epochs, min((bi+1)*batch_sz, total_win), total_win, "training"), U
        scheduler.step(); ep_loss = rl / len(train_ds)

        f1m, mAP, f1p, app = 0.0, 0.0, [], []
        if val_loader:
            model.eval(); window_probs = []
            with torch.no_grad():
                for vi, (vids, _) in enumerate(val_loader):
                    vids = vids.to(device)
                    with autocast(): pr = torch.softmax(model(vids), dim=1).cpu().numpy()
                    window_probs.extend(pr)
                    if (vi + 1) % 5 == 0 or vi == len(val_loader) - 1:
                        yield html_progress(ep, n_epochs, min((vi+1)*batch_sz, len(val_ds)), len(val_ds), "validating"), U
            metrics = compute_framewise_metrics(np.array(window_probs), val_ds.frame_mappings if hasattr(val_ds, 'frame_mappings') else [], new_nc)
            f1m = metrics["f1"]; mAP = metrics["mAP"]; f1p = metrics["f1_per"]; app = metrics["ap_per"]

        mp = os.path.join(odir, f"epoch_{ep+1}_f1_{f1m:.4f}_map_{mAP:.4f}.pth")
        torch.save(model.state_dict(), mp)
        S["train_log"].append({"epoch": ep+1, "loss": ep_loss, "f1": f1m, "mAP": mAP, "f1_per": f1p, "ap_per": app, "path": mp})
        yield html_progress(ep+1, n_epochs, total_win, total_win, "done"), build_val_html(S["train_log"], new_names)

    import json
    with open(os.path.join(odir, "training_log.json"), "w") as f:
        json.dump(S["train_log"], f, indent=2)
    print("✅ Training complete!")

# ====================== Cursor JS ======================

CURSOR_JS = """
(fi, labels_json) => {
    let T=0, names=[], labels=[];
    try { const d=JSON.parse(labels_json); T=d.T; names=d.names; labels=d.labels; } catch(e) { return fi; }
    if (T===0) return fi;
    fi = Math.max(0, Math.min(Math.floor(fi), T-1));
    const c = document.getElementById('tl-cursor');
    if (c) c.style.left = ((fi/T)*100)+'%';
    const l = document.getElementById('tl-frame-label');
    if (l) { const cls=labels[fi]; l.textContent='F:'+fi+' '+(names[cls]||'?'); }
    return fi;
}
"""

# ====================== GUI ======================

YELLOW_THEME = gr.themes.Soft(
    primary_hue=gr.themes.colors.amber,
    secondary_hue=gr.themes.colors.yellow,
    neutral_hue=gr.themes.colors.gray,
)

with gr.Blocks(title="Animal Behavior Training", theme=YELLOW_THEME) as demo:
    gr.Markdown("# 🏋️ Animal Behavior Model Training")
    gr.Markdown("Fine-tune from pretrained — preview labels & configure mapping before training")

    cursor_state = gr.Textbox(value=S["_cursor_data"], visible=False)
    repo_in = gr.Textbox(value=HF_REPO_ID, visible=False)

    with gr.Row():
        with gr.Column(scale=1, min_width=250):
            gr.Markdown("### ① Select model")
            model_dd = gr.Dropdown(label="Base model", choices=[], interactive=True)
            load_btn = gr.Button("📥 Load pretrained", variant="primary")
            model_st = gr.Textbox(label="Status", interactive=False, lines=4)
            gr.Markdown("---")
            gr.Markdown("### ② Load data")
            vdir_in = gr.Textbox(label="Video directory", value=DEFAULT_VIDEO_DIR)
            ldir_in = gr.Textbox(label="Label directory", value=DEFAULT_LABEL_DIR)
            odir_in = gr.Textbox(label="Output directory", value=DEFAULT_OUTPUT_DIR)
            scan_d = gr.Button("📂 Load folder", variant="secondary")
            scan_st = gr.Textbox(label="Folder status", interactive=False, lines=1)

        with gr.Column(scale=2, min_width=400):
            progress_html = gr.HTML("")
            info_html = gr.HTML("<p style='color:#aaa;'>Load data to preview</p>")
            frame_img = gr.Image(label="Frame preview", type="numpy", interactive=False)
            timeline_html = gr.HTML("")
            scrubber = gr.Slider(minimum=0, maximum=100, step=1, value=0, label="Frame", interactive=True)
            with gr.Row():
                prev_btn = gr.Button("◀ Prev", size="sm")
                nav_md = gr.Markdown("*Load data first*")
                next_btn = gr.Button("Next ▶", size="sm")
            gr.Markdown("---")
            with gr.Accordion("📹 Videos", open=False):
                vid_list_html = gr.HTML("<p style='color:#aaa;font-size:12px;'>Load data first</p>")
            with gr.Accordion("📊 Label distribution", open=False):
                label_dist_html = gr.HTML("<p style='color:#aaa;'>Load data to see labels</p>")
            gr.Markdown("### 🏷️ Label mapping")
            head_mode_dd = gr.Dropdown(label="Head type", choices=["Pretrain head", "New head"], value="Pretrain head", interactive=True)
            map_dds = []
            for i in range(MAX_LABELS):
                dd = gr.Dropdown(label=f"label_{i}", choices=[], value=None, interactive=True, visible=False)
                map_dds.append(dd)
            mapping_summary = gr.HTML("")
            vid_dd = gr.Dropdown(label="Video", choices=[], interactive=True, visible=False)

        with gr.Column(scale=1, min_width=280):
            gr.Markdown("### ③ Train")
            vr_in = gr.Slider(minimum=0, maximum=50, step=5, value=15, label="Validation ratio (%)", interactive=True)
            with gr.Row():
                ep_in = gr.Number(label="Epochs", value=5, precision=0)
                bs_in = gr.Number(label="Batch", value=8, precision=0)
            lr_in = gr.Textbox(label="LR", value="3.8e-5")
            with gr.Row():
                ws_in = gr.Number(label="Window", value=16, precision=0)
                st_in = gr.Number(label="Stride", value=4, precision=0)
            with gr.Row():
                val_seed_in = gr.Number(label="Val seed", value=1337, precision=0)
                train_seed_in = gr.Number(label="Train seed", value=2025, precision=0)
            train_btn = gr.Button("🚀 Start training", variant="primary", size="lg")
            gr.Markdown("---")
            gr.Markdown("### ④ Validation results")
            val_html = gr.HTML("<p style='color:#aaa;'>Training not started</p>")

    # ===== WIRING =====
    demo.load(list_models, [repo_in], [model_dd, model_st])
    load_btn.click(load_pretrained, [repo_in, model_dd], [model_st])

    scan_outputs = [scan_st, label_dist_html, nav_md, *map_dds, vid_dd,
                    frame_img, info_html, timeline_html, scrubber, cursor_state, vid_list_html, mapping_summary]
    scan_d.click(do_scan_and_preview, [vdir_in, ldir_in, vr_in, val_seed_in, head_mode_dd, *map_dds], scan_outputs)

    map_change_outputs = [*map_dds, timeline_html, cursor_state, mapping_summary]
    head_mode_dd.change(on_mapping_change, [head_mode_dd, *map_dds], map_change_outputs)
    for dd in map_dds:
        dd.change(on_mapping_change, [head_mode_dd, *map_dds], map_change_outputs)

    vr_in.change(on_val_ratio_change, [vr_in, val_seed_in], [vid_list_html])
    val_seed_in.change(on_val_ratio_change, [vr_in, val_seed_in], [vid_list_html])
    vid_dd.change(on_vid_change, [vid_dd, head_mode_dd, *map_dds], [frame_img, info_html, timeline_html, scrubber, cursor_state, vid_list_html, nav_md])
    scrubber.input(fn=None, inputs=[scrubber, cursor_state], outputs=[scrubber], js=CURSOR_JS)
    scrubber.change(on_scrub, [scrubber, head_mode_dd, *map_dds], [frame_img, info_html])
    prev_btn.click(lambda hm, *dd: do_nav("prev", hm, *dd), [head_mode_dd, *map_dds], [frame_img, info_html, timeline_html, scrubber, cursor_state, nav_md, vid_list_html])
    next_btn.click(lambda hm, *dd: do_nav("next", hm, *dd), [head_mode_dd, *map_dds], [frame_img, info_html, timeline_html, scrubber, cursor_state, nav_md, vid_list_html])
    train_btn.click(run_training, [repo_in, model_dd, vdir_in, ldir_in, odir_in, head_mode_dd, ep_in, bs_in, lr_in, vr_in, ws_in, st_in, val_seed_in, train_seed_in, *map_dds], [progress_html, val_html])

if __name__ == "__main__":
    demo.launch(debug=True, share=True)
