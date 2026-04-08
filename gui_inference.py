"""
Gradio GUI for animal behavior inference.

Usage:
    python gui_inference.py
"""

import os, json
import numpy as np
import torch
import gradio as gr
import pandas as pd
from PIL import Image
from collections import Counter
from decord import VideoReader, cpu
from huggingface_hub import hf_hub_download, list_repo_files

from models import build_model_from_config
from config_utils import normalize_config, find_config_for_pth
from inference import preprocess, infer_video_gen, remap_with_disabled, get_others_idx

# ==================== 👇 修改這裡 👇 ====================
HF_REPO_ID         = "yiheng266/animal-social-models"
DEFAULT_VIDEO_DIR  = "/content/drive/My Drive/videos/"
DEFAULT_OUTPUT_DIR = "/content/drive/My Drive/results/"
DEFAULT_LOCAL_MODEL_DIRS = []
# ==================== 👆 修改以上即可 👆 ====================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ====================== State ======================

S = {"model": None, "cfg": None, "results": {}, "cur": None, "vr": None,
     "done": [], "idx": 0, "_active_vdir": None,
     "_cursor_data": json.dumps({"T": 0, "names": [], "labels": []}),
     "model_source": None, "disabled_classes": set()}

CLR_PALETTE    = ["#378ADD","#D85A30","#E24B4A","#7F77DD","#1D9E75","#BA7517","#888780"]
CLR_BG_PALETTE = ["rgba(55,138,221,0.9)","rgba(216,90,48,0.9)","rgba(226,75,74,0.9)",
                   "rgba(127,119,221,0.9)","rgba(29,158,117,0.9)","rgba(186,117,23,0.9)","rgba(136,135,128,0.9)"]

def get_clr(cfg, i):
    names = cfg["class_names"]
    nm = names[i] if i < len(names) else "others"
    if nm.lower() in ("others", "other"):
        return "#FFFFFF", "rgba(180,180,180,0.9)"
    idx = i % len(CLR_PALETTE)
    return CLR_PALETTE[idx], CLR_BG_PALETTE[idx]

U = gr.update()

# ====================== Model Loading ======================

def list_models(repo):
    try:
        files = list_repo_files(repo)
        pth_files = [f for f in files if f.endswith("/model.pth") or f == "model.pth"]
        if not pth_files:
            pth_files = [f for f in files if f.endswith(".pth")]
        if not pth_files:
            return gr.update(choices=[], value=None), "❌ No models found"
        model_names = [os.path.dirname(p) if "/" in p else p for p in pth_files]
        model_names = [m for m in model_names if not m.startswith("k400_")]
        return gr.update(choices=model_names, value=model_names[0] if model_names else None), f"✅ {len(model_names)} model(s)"
    except Exception as e:
        return gr.update(choices=[], value=None), f"❌ {e}"

def _post_load_updates(cfg):
    names = cfg["class_names"]
    toggle_choices = [nm for nm in names if nm.lower() not in ("others", "other")]
    S["disabled_classes"] = set()
    return (
        gr.update(choices=toggle_choices, value=toggle_choices, visible=True),
        _html_behavior_toggles(cfg),
    )

def _html_behavior_toggles(cfg):
    if not cfg:
        return "<p style='color:#aaa;font-size:13px;'>Load a model first</p>"
    names = cfg["class_names"]
    items = ""
    for i, nm in enumerate(names):
        if nm.lower() in ("others", "other"):
            continue
        _, bg = get_clr(cfg, i)
        items += (f"<span style='display:inline-flex;align-items:center;gap:4px;margin-right:6px;"
                  f"padding:3px 10px;border-radius:12px;background:{bg};color:white;"
                  f"font-size:12px;font-weight:600;'>{nm}</span>")
    return f"<div style='display:flex;flex-wrap:wrap;gap:4px;align-items:center;'>{items}</div>"

def load_model_hf(repo, model_name):
    if not model_name or not repo:
        return "❌ Specify repo & model", U, ""
    try:
        if "/" in model_name or not model_name.endswith(".pth"):
            cfg_file = f"{model_name}/config.json"
            pth_file = f"{model_name}/model.pth"
        else:
            cfg_file = "config.json"
            pth_file = model_name
        with open(hf_hub_download(repo_id=repo, filename=cfg_file)) as f:
            raw = json.load(f)
        cfg, err = normalize_config(raw)
        if err:
            return f"❌ {err}", U, ""
        pth_path = hf_hub_download(repo_id=repo, filename=pth_file)
        model = build_model_from_config(cfg)
        model.load_state_dict(torch.load(pth_path, map_location=device, weights_only=True))
        model.to(device).eval()
        S.update({"model": model, "cfg": cfg, "results": {}, "done": [], "cur": None, "vr": None, "model_source": "hf"})
        toggle_upd, toggle_html = _post_load_updates(cfg)
        return (f"✅ Loaded from HuggingFace!\n"
                f"  Model: {model_name}\n"
                f"  Backbone: {cfg['backbone']['name']} | Frames: {cfg['backbone']['num_frames']}\n"
                f"  Classes ({len(cfg['class_names'])}): {cfg['class_names']}\n"
                f"  Device: {device}",
                toggle_upd, toggle_html)
    except Exception as e:
        return f"❌ {e}", U, ""

def load_model_local(local_dir, pth_name):
    if not pth_name or not local_dir:
        return "❌ Select a local model", U, ""
    pth_path = os.path.join(local_dir, pth_name)
    if not os.path.exists(pth_path):
        return f"❌ File not found: {pth_path}", U, ""
    try:
        cfg, cfg_source, err = find_config_for_pth(pth_path)
        if err:
            return f"❌ {err}\n\n💡 Place a config.json or <name>_config.json in same folder.", U, ""
        model = build_model_from_config(cfg)
        model.load_state_dict(torch.load(pth_path, map_location=device, weights_only=True))
        model.to(device).eval()
        S.update({"model": model, "cfg": cfg, "results": {}, "done": [], "cur": None, "vr": None, "model_source": "local"})
        toggle_upd, toggle_html = _post_load_updates(cfg)
        size_mb = os.path.getsize(pth_path) / 1024**2
        return (f"✅ Loaded local model!\n"
                f"  File: {pth_name} ({size_mb:.1f} MB)\n"
                f"  Config: {cfg_source}\n"
                f"  Backbone: {cfg['backbone']['name']} | Frames: {cfg['backbone']['num_frames']}\n"
                f"  Classes ({len(cfg['class_names'])}): {cfg['class_names']}\n"
                f"  Device: {device}",
                toggle_upd, toggle_html)
    except Exception as e:
        import traceback
        return f"❌ Load failed: {e}\n\n{traceback.format_exc()}", U, ""

def scan_local_models(local_dir):
    if not local_dir or not os.path.isdir(local_dir):
        return gr.update(choices=[], value=None), "❌ Folder not found"
    pth_files = []
    for f in sorted(os.listdir(local_dir)):
        fp = os.path.join(local_dir, f)
        if f.endswith(".pth") and os.path.isfile(fp):
            pth_files.append(f)
        elif os.path.isdir(fp):
            for sf in sorted(os.listdir(fp)):
                if sf.endswith(".pth"):
                    pth_files.append(os.path.join(f, sf))
    if not pth_files:
        return gr.update(choices=[], value=None), "❌ No .pth files found"
    return gr.update(choices=pth_files, value=pth_files[0]), f"✅ {len(pth_files)} model(s) found"

def on_toggle_change(enabled_behaviors):
    if not S["cfg"]:
        return ""
    names = S["cfg"]["class_names"]
    disabled = set()
    for i, nm in enumerate(names):
        if nm.lower() in ("others", "other"):
            continue
        if nm not in enabled_behaviors:
            disabled.add(i)
    S["disabled_classes"] = disabled
    if disabled:
        return f"⚠️ Disabled → Other: {[names[i] for i in sorted(disabled)]}"
    return "✅ All behaviors active"

DEMO_LOCAL_DIR = os.path.join(os.path.expanduser("~"), "demo_data")

def load_demo_inference(repo):
    """Download all files from HF demo/, scan videos, preview first one.
    Does NOT modify the video folder path textbox."""
    if not repo:
        return gr.update(choices=[], value=None), "❌ Specify repo", None, "<p style='color:#aaa;'>Select a video</p>", gr.update(maximum=0, value=0), "", S["_cursor_data"]
    try:
        all_files = list_repo_files(repo)
        demo_files = [f for f in all_files if f.startswith("demo/") and f != "demo/"]
        if not demo_files:
            return gr.update(choices=[], value=None), "❌ No files in demo/ folder", None, "", gr.update(maximum=0, value=0), "", S["_cursor_data"]
        os.makedirs(DEMO_LOCAL_DIR, exist_ok=True)
        for f in demo_files:
            local = hf_hub_download(repo_id=repo, filename=f)
            fname = os.path.basename(f)
            dest = os.path.join(DEMO_LOCAL_DIR, fname)
            if not os.path.exists(dest) or os.path.getsize(dest) != os.path.getsize(local):
                import shutil; shutil.copy2(local, dest)
        # Scan for videos
        videos = sorted([f for f in os.listdir(DEMO_LOCAL_DIR) if f.lower().endswith((".mp4", ".avi", ".mov"))])
        if not videos:
            return gr.update(choices=[], value=None), "⚠️ No videos in demo/", None, "", gr.update(maximum=0, value=0), "", S["_cursor_data"]
        # Store active dir in state
        S["_active_vdir"] = DEMO_LOCAL_DIR
        # Preview first video
        vf = videos[0]
        vp = os.path.join(DEMO_LOCAL_DIR, vf)
        try:
            vr = VideoReader(vp, ctx=cpu(0))
            S["_preview_vr"] = vr; S["_preview_vf"] = vf
            T = len(vr); fps = vr.get_avg_fps()
            info = (f"<div style='display:flex;justify-content:space-between;align-items:center;'>"
                    f"<span style='padding:4px 12px;border-radius:6px;background:rgba(180,180,180,0.7);color:white;font-size:13px;font-weight:600;'>Preview</span>"
                    f"<span style='font-size:12px;color:#666;'>F: 0/{T} | 0.00s/{T/fps:.2f}s</span></div>")
            img = vr[0].asnumpy()
        except:
            T = 0; info = ""; img = None
        return (gr.update(choices=videos, value=vf),
                f"✅ Demo loaded: {len(videos)} video(s)",
                img, info, gr.update(maximum=max(T - 1, 0), value=0), "", S["_cursor_data"])
    except Exception as e:
        return gr.update(choices=[], value=None), f"❌ {e}", None, "", gr.update(maximum=0, value=0), "", S["_cursor_data"]

def scan_videos_and_preview(vdir):
    """Load folder: scan videos and preview the first one."""
    if not vdir or not os.path.isdir(vdir):
        return gr.update(choices=[], value=None), "❌ Not found", None, "", gr.update(maximum=0, value=0), "", S["_cursor_data"]
    v = sorted([f for f in os.listdir(vdir) if f.lower().endswith((".mp4", ".avi", ".mov"))])
    if not v:
        return gr.update(choices=[], value=None), "❌ No videos", None, "", gr.update(maximum=0, value=0), "", S["_cursor_data"]
    S["_active_vdir"] = vdir
    # Preview first video
    vf = v[0]
    vp = os.path.join(vdir, vf)
    try:
        vr = VideoReader(vp, ctx=cpu(0))
        S["_preview_vr"] = vr; S["_preview_vf"] = vf
        T = len(vr); fps = vr.get_avg_fps()
        info = (f"<div style='display:flex;justify-content:space-between;align-items:center;'>"
                f"<span style='padding:4px 12px;border-radius:6px;background:rgba(180,180,180,0.7);color:white;font-size:13px;font-weight:600;'>Preview</span>"
                f"<span style='font-size:12px;color:#666;'>F: 0/{T} | 0.00s/{T/fps:.2f}s</span></div>")
        img = vr[0].asnumpy()
    except:
        T = 0; info = ""; img = None
    return (gr.update(choices=v, value=vf), f"✅ {len(v)} videos",
            img, info, gr.update(maximum=max(T - 1, 0), value=0), "", S["_cursor_data"])

# ====================== HTML Builders ======================

def html_progress(vd, vt, cur_name, wd, wt):
    if vt == 0: return ""
    vp = (vd / vt) * 100; wp = (wd / max(wt, 1)) * 100
    vc = "#1D9E75" if vd == vt else "#D85A30"
    st = "✅ Complete" if vd == vt else "Processing..."
    return f"""<div style="background:#fff;border:1px solid #e0e0e0;border-radius:8px;padding:10px 14px;">
      <div style="display:flex;justify-content:space-between;margin-bottom:4px;">
        <span style="font-size:13px;font-weight:600;">Batch — {st}</span>
        <span style="font-size:12px;color:#888;">{vd}/{vt} videos</span></div>
      <div style="height:8px;background:#eee;border-radius:4px;overflow:hidden;margin-bottom:10px;">
        <div style="width:{vp:.1f}%;height:100%;background:{vc};border-radius:4px;"></div></div>
      <div style="display:flex;justify-content:space-between;margin-bottom:4px;">
        <span style="font-size:12px;font-weight:500;">Current: {cur_name}</span>
        <span style="font-size:12px;color:#888;">{wd}/{wt} windows</span></div>
      <div style="height:6px;background:#eee;border-radius:3px;overflow:hidden;">
        <div style="width:{wp:.1f}%;height:100%;background:#1D9E75;border-radius:3px;"></div></div>
    </div>"""

def html_timeline(vf):
    r = S["results"].get(vf)
    if not r: return ""
    cfg = S["cfg"]; names = cfg["class_names"]; labels = r["frame_labels"]; T = len(labels)
    if T == 0: return ""
    segs = []; cur, cnt = labels[0], 1
    for i in range(1, T):
        if labels[i] == cur: cnt += 1
        else: segs.append((cur, cnt)); cur, cnt = labels[i], 1
    segs.append((cur, cnt))
    bar = ""
    for li, c in segs:
        clr, _ = get_clr(cfg, li); pct = (c / T) * 100
        bdr = "border-top:1px solid #ccc;border-bottom:1px solid #ccc;" if clr == "#FFFFFF" else ""
        nm = names[li] if li < len(names) else "?"
        bar += f"<div style='width:{pct:.3f}%;height:100%;background:{clr};{bdr}display:inline-block;box-sizing:border-box;' title='{nm}'></div>"
    leg = ""
    for i, nm in enumerate(names):
        clr, _ = get_clr(cfg, i)
        bdr = "border:1px solid #ccc;" if clr == "#FFFFFF" else ""
        leg += (f"<span style='display:inline-flex;align-items:center;gap:4px;margin-right:12px;font-size:12px;'>"
                f"<span style='display:inline-block;width:10px;height:10px;border-radius:2px;background:{clr};{bdr}'></span>{nm}</span>")
    nm0 = names[labels[0]] if labels[0] < len(names) else "?"
    return f"""<div style="width:100%;padding:4px 0;">
      <div style="position:relative;display:flex;height:18px;border-radius:4px;overflow:hidden;border:1px solid #ccc;">
        {bar}
        <div id="tl-cursor" style="position:absolute;top:-2px;bottom:-2px;width:2px;background:#000;left:0%;pointer-events:none;"></div>
      </div>
      <div style="display:flex;justify-content:space-between;align-items:center;margin-top:4px;">
        <div>{leg}</div>
        <span id="tl-frame-label" style="font-size:12px;font-weight:500;color:#555;">F:0 {nm0}</span>
      </div></div>"""

def html_behavior(vf):
    r = S["results"].get(vf)
    if not r: return "<p style='color:#aaa;'>Run inference first</p>"
    cfg = S["cfg"]; names = cfg["class_names"]; T = r["total_frames"]; fps = r["fps"]
    cnt = Counter(r["frame_labels"])
    h = f"<div style='padding:4px 0;'>"
    h += f"<div style='display:flex;justify-content:space-between;margin-bottom:8px;'><span style='font-size:14px;font-weight:600;'>Behavior statistics</span><span style='font-size:12px;color:#888;'>{vf}</span></div>"
    for i, nm in enumerate(names):
        c = cnt.get(i, 0); pct = 100 * c / max(T, 1); dur = c / max(fps, 1)
        clr, _ = get_clr(cfg, i); bar_clr = "#ddd" if clr == "#FFFFFF" else clr
        h += f"<div style='margin-bottom:8px;'><div style='display:flex;justify-content:space-between;margin-bottom:2px;'><span style='font-size:13px;font-weight:500;'>{nm}</span><span style='font-size:12px;color:#888;'>{c:,} fr · {pct:.1f}% · {dur:.1f}s</span></div><div style='height:8px;background:#f0f0f0;border-radius:4px;overflow:hidden;'><div style='width:{max(pct,0.3):.1f}%;height:100%;background:{bar_clr};border-radius:4px;'></div></div></div>"
    h += f"<div style='display:flex;gap:10px;margin-top:10px;padding-top:8px;border-top:1px solid #eee;'><div style='flex:1;background:#f7f7f7;border-radius:6px;padding:6px;text-align:center;'><div style='font-size:11px;color:#888;'>Frames</div><div style='font-size:16px;font-weight:600;'>{T:,}</div></div><div style='flex:1;background:#f7f7f7;border-radius:6px;padding:6px;text-align:center;'><div style='font-size:11px;color:#888;'>FPS</div><div style='font-size:16px;font-weight:600;'>{fps:.1f}</div></div><div style='flex:1;background:#f7f7f7;border-radius:6px;padding:6px;text-align:center;'><div style='font-size:11px;color:#888;'>Duration</div><div style='font-size:16px;font-weight:600;'>{T/fps:.1f}s</div></div></div></div>"
    return h

def html_export_preview(vf, fmt):
    r = S["results"].get(vf)
    if not r: return "<p style='color:#aaa;font-size:13px;'>Run inference first</p>"
    names = S["cfg"]["class_names"]; labels = r["frame_labels"]; fps = r["fps"]
    td = "padding:3px 6px;border-bottom:1px solid #eee;font-size:11px;font-family:monospace;"
    th = f"{td}font-weight:bold;color:#666;"
    if fmt == "One-hot CSV (per-frame)":
        hdr = f"<tr><td style='{th}'>frame</td>" + "".join(f"<td style='{th}'>{n[:6]}</td>" for n in names) + "</tr>"
        rows = "".join(
            f"<tr><td style='{td}'>{i}</td>" + "".join(f"<td style='{td}'>{'1' if labels[i]==ci else '0'}</td>" for ci in range(len(names))) + "</tr>"
            for i in range(min(5, len(labels)))
        ) + f"<tr><td style='{td}' colspan='{len(names)+1}'>... ({len(labels)} rows)</td></tr>"
        title = "One-hot CSV preview"
    else:
        hdr = f"<tr><td style='{th}'>time</td><td style='{th}'>media</td><td style='{th}'>subj</td><td style='{th}'>behavior</td><td style='{th}'>status</td></tr>"
        evts = []
        if labels:
            cur, st = labels[0], 0
            for i in range(1, len(labels)):
                if labels[i] != cur: evts.append((st, i, cur)); cur, st = labels[i], i
            evts.append((st, len(labels), cur))
        rows = "".join(
            f"<tr><td style='{td}'>{s/fps:.3f}</td><td style='{td}'>{vf[:12]}</td><td style='{td}'>pair</td><td style='{td}'>{names[c] if c<len(names) else '?'}</td><td style='{td}'>START</td></tr>"
            f"<tr><td style='{td}'>{e/fps:.3f}</td><td style='{td}'>{vf[:12]}</td><td style='{td}'>pair</td><td style='{td}'>{names[c] if c<len(names) else '?'}</td><td style='{td}'>STOP</td></tr>"
            for s, e, c in evts[:3]
        ) + f"<tr><td style='{td}' colspan='5'>... ({len(evts)} events)</td></tr>"
        title = "BORIS event log preview"
    return f"<div style='margin-top:4px;'><p style='font-size:12px;color:#666;font-weight:600;margin:0 0 4px;'>{title}</p><div style='overflow-x:auto;border:1px solid #eee;border-radius:4px;'><table style='border-collapse:collapse;width:100%;'>{hdr}{rows}</table></div></div>"

def update_export_preview(fmt):
    return html_export_preview(S["cur"], fmt) if S["cur"] else "<p style='color:#aaa;'>No results</p>"

# ====================== Display ======================

def preview_frame(vdir, vf, fi):
    """Get a frame from a video file directly (no inference needed)."""
    if not vf or not vdir:
        return None
    vp = os.path.join(vdir, vf)
    if not os.path.exists(vp):
        return None
    try:
        if S.get("_preview_vf") != vf or S.get("_preview_vr") is None:
            S["_preview_vr"] = VideoReader(vp, ctx=cpu(0))
            S["_preview_vf"] = vf
        vr = S["_preview_vr"]
        fi = max(0, min(int(fi), len(vr) - 1))
        return vr[fi].asnumpy()
    except:
        return None

def preview_info_html(vdir, vf, fi):
    """Frame info for preview mode (no inference labels)."""
    if not vf or not vdir:
        return "<p style='color:#aaa;'>Select a video to preview</p>"
    # If we have inference results, use those
    r = S["results"].get(vf)
    if r:
        return frame_info_html(vf, fi)
    # Otherwise just show frame / time info
    vp = os.path.join(vdir, vf)
    if not os.path.exists(vp):
        return "<p style='color:#aaa;'>Video not found</p>"
    try:
        if S.get("_preview_vf") != vf or S.get("_preview_vr") is None:
            S["_preview_vr"] = VideoReader(vp, ctx=cpu(0))
            S["_preview_vf"] = vf
        vr = S["_preview_vr"]
        T = len(vr); fps = vr.get_avg_fps(); fi = max(0, min(int(fi), T - 1))
        return (f"<div style='display:flex;justify-content:space-between;align-items:center;'>"
                f"<span style='padding:4px 12px;border-radius:6px;background:rgba(180,180,180,0.7);color:white;font-size:13px;font-weight:600;'>Preview</span>"
                f"<span style='font-size:12px;color:#666;'>F: {fi}/{T} | {fi/fps:.2f}s/{T/fps:.2f}s</span></div>")
    except:
        return "<p style='color:#aaa;'>Cannot read video</p>"

def _vdir(vdir_input):
    """Return active video dir: prefer S state (set by demo/load), fallback to textbox."""
    return S.get("_active_vdir") or vdir_input

def on_video_select(vf):
    """When user selects a video from dropdown, show first frame + set scrubber."""
    vdir = S.get("_active_vdir")
    if not vf or not vdir:
        return None, "<p style='color:#aaa;'>Select a video</p>", gr.update(maximum=0, value=0), "", S["_cursor_data"]
    # If we have inference results for this video, show full view
    r = S["results"].get(vf)
    if r:
        S["cur"] = vf; S["vr"] = None; _update_cursor(vf)
        T = r["total_frames"]
        return (get_frame(vf, 0), frame_info_html(vf, 0),
                gr.update(maximum=max(T - 1, 0), value=0),
                html_timeline(vf), S["_cursor_data"])
    # No results yet — just preview raw video
    vp = os.path.join(vdir, vf)
    if not os.path.exists(vp):
        return None, "<p style='color:#aaa;'>Video not found</p>", gr.update(maximum=0, value=0), "", S["_cursor_data"]
    try:
        vr = VideoReader(vp, ctx=cpu(0))
        S["_preview_vr"] = vr; S["_preview_vf"] = vf
        T = len(vr); fps = vr.get_avg_fps()
        info = (f"<div style='display:flex;justify-content:space-between;align-items:center;'>"
                f"<span style='padding:4px 12px;border-radius:6px;background:rgba(180,180,180,0.7);color:white;font-size:13px;font-weight:600;'>Preview</span>"
                f"<span style='font-size:12px;color:#666;'>F: 0/{T} | 0.00s/{T/fps:.2f}s</span></div>")
        return vr[0].asnumpy(), info, gr.update(maximum=max(T - 1, 0), value=0), "", S["_cursor_data"]
    except Exception as e:
        return None, f"<p style='color:#aaa;'>Error: {e}</p>", gr.update(maximum=0, value=0), "", S["_cursor_data"]

def get_frame(vf, fi):
    r = S["results"].get(vf)
    if not r: return None
    if S["cur"] != vf or S["vr"] is None:
        S["vr"] = VideoReader(r["video_path"], ctx=cpu(0)); S["cur"] = vf
    return S["vr"][max(0, min(fi, len(S["vr"]) - 1))].asnumpy()

def frame_info_html(vf, fi):
    r = S["results"].get(vf)
    if not r: return "<p style='color:#aaa;'>Run inference first</p>"
    cfg = S["cfg"]; names = cfg["class_names"]; T = r["total_frames"]; fps = r["fps"]
    fi = max(0, min(fi, T - 1)); li = r["frame_labels"][fi]; nm = names[li]
    conf = r["frame_confidences"][fi][li] * 100; _, bg = get_clr(cfg, li)
    return (f"<div style='display:flex;justify-content:space-between;align-items:center;'>"
            f"<span style='padding:4px 12px;border-radius:6px;background:{bg};color:white;font-size:13px;font-weight:600;'>{nm} ({conf:.0f}%)</span>"
            f"<span style='font-size:12px;color:#666;'>F: {fi}/{T} | {fi/fps:.2f}s/{T/fps:.2f}s</span></div>")

def nav_md():
    d = S["done"]; i = S["idx"]
    if not d: return "*No results*"
    return f"**{d[i]}** — {i+1}/{len(d)} completed"

def _update_cursor(vf):
    r = S["results"].get(vf)
    if not r: S["_cursor_data"] = json.dumps({"T": 0, "names": [], "labels": []})
    else: S["_cursor_data"] = json.dumps({"T": r["total_frames"], "names": S["cfg"]["class_names"], "labels": r["frame_labels"]})

def _full(vf, fi=0, vd=0, vt=0):
    r = S["results"].get(vf)
    if not r:
        e = ""; return e, e, None, e, e, e, "*No results*", gr.update(maximum=0, value=0), S["_cursor_data"]
    S["cur"] = vf; S["vr"] = None
    if vf in S["done"]: S["idx"] = S["done"].index(vf)
    T = r["total_frames"]; _update_cursor(vf)
    return (html_progress(vd, vt, vf, T, T), frame_info_html(vf, fi), get_frame(vf, fi),
            html_timeline(vf), html_behavior(vf),
            html_export_preview(vf, "One-hot CSV (per-frame)"),
            nav_md(), gr.update(maximum=max(T - 1, 0), value=0), S["_cursor_data"])

# ====================== Actions ======================

def run_single(vf):
    vdir = S.get("_active_vdir")
    if not S["model"]: yield "", "", None, "", "", "", "❌ Load model first", U, S["_cursor_data"]; return
    if not vf or not vdir: yield "", "", None, "", "", "", "❌ Select video", U, S["_cursor_data"]; return
    result = None
    for msg in infer_video_gen(vdir, vf, S["model"], S["cfg"], S["disabled_classes"]):
        if isinstance(msg, dict): result = msg
        else:
            wd, wt = msg
            yield html_progress(0, 1, vf, wd, wt), U, U, U, U, U, U, U, U
    S["results"][vf] = result
    if vf not in S["done"]: S["done"].append(vf)
    yield _full(vf, 0, 1, 1)

def run_batch():
    vdir = S.get("_active_vdir")
    if not S["model"]: yield "", "", None, "", "", "", "❌ Load model first", U, S["_cursor_data"], ""; return
    if not vdir or not os.path.isdir(vdir): yield "", "", None, "", "", "", "❌ Load videos first", U, S["_cursor_data"], ""; return
    vids = sorted([f for f in os.listdir(vdir) if f.lower().endswith((".mp4", ".avi", ".mov"))])
    if not vids: yield "", "", None, "", "", "", "❌ No videos", U, S["_cursor_data"], ""; return
    total = len(vids); blog = []
    for vi, vf in enumerate(vids):
        result = None
        for msg in infer_video_gen(vdir, vf, S["model"], S["cfg"], S["disabled_classes"]):
            if isinstance(msg, dict): result = msg
            else:
                wd, wt = msg
                yield html_progress(vi, total, vf, wd, wt), U, U, U, U, U, U, U, U, U
        S["results"][vf] = result
        if vf not in S["done"]: S["done"].append(vf)
        blog.append(f"✅ {vf} ({result['total_frames']} fr)")
        S["cur"] = vf; S["vr"] = None; _update_cursor(vf)
        if vf in S["done"]: S["idx"] = S["done"].index(vf)
        T = result["total_frames"]
        yield (html_progress(vi + 1, total, vf, T, T),
               frame_info_html(vf, 0), get_frame(vf, 0),
               html_timeline(vf), html_behavior(vf),
               html_export_preview(vf, "One-hot CSV (per-frame)"),
               nav_md(), gr.update(maximum=max(T - 1, 0), value=0),
               S["_cursor_data"], "\n".join(blog))

def on_scrub(fi):
    fi = int(fi)
    vdir = S.get("_active_vdir")
    vf = S["cur"]
    # If inference results exist, use them
    if vf and vf in S["results"]:
        return get_frame(vf, fi), frame_info_html(vf, fi)
    # Otherwise, preview mode — use the currently selected video
    preview_vf = S.get("_preview_vf")
    if preview_vf and vdir:
        return preview_frame(vdir, preview_vf, fi), preview_info_html(vdir, preview_vf, fi)
    return None, "<p style='color:#aaa;'>Select a video to preview</p>"

def do_nav(direction):
    d = S["done"]
    if not d: return "", "", None, "", "", "", "*No results*", gr.update(), S["_cursor_data"]
    if direction == "prev": S["idx"] = max(0, S["idx"] - 1)
    else: S["idx"] = min(len(d) - 1, S["idx"] + 1)
    return _full(d[S["idx"]], 0, len(d), len(d))

# ====================== Export ======================

def _exp_onehot(vf, od):
    if vf not in S["results"]: return "❌"
    r = S["results"][vf]; names = S["cfg"]["class_names"]; nc = len(names)
    os.makedirs(od, exist_ok=True)
    rows = [[1 if l == ci else 0 for ci in range(nc)] for l in r["frame_labels"]]
    df = pd.DataFrame(rows, columns=names); df.insert(0, "frame", range(len(rows)))
    p = os.path.join(od, vf.rsplit(".", 1)[0] + "_onehot.csv"); df.to_csv(p, index=False)
    return f"✅ {p}"

def _exp_boris(vf, od):
    if vf not in S["results"]: return "❌"
    r = S["results"][vf]; names = S["cfg"]["class_names"]; fps = r["fps"]; L = r["frame_labels"]
    os.makedirs(od, exist_ok=True); evts = []; cur, st = L[0], 0
    for i in range(1, len(L)):
        if L[i] != cur:
            evts += [{"Time": round(st/fps, 3), "Media": vf, "Subject": "pair", "Behavior": names[cur], "Status": "START"},
                     {"Time": round(i/fps, 3),  "Media": vf, "Subject": "pair", "Behavior": names[cur], "Status": "STOP"}]
            cur, st = L[i], i
    evts += [{"Time": round(st/fps, 3),       "Media": vf, "Subject": "pair", "Behavior": names[cur], "Status": "START"},
             {"Time": round(len(L)/fps, 3),    "Media": vf, "Subject": "pair", "Behavior": names[cur], "Status": "STOP"}]
    p = os.path.join(od, vf.rsplit(".", 1)[0] + "_boris.csv"); pd.DataFrame(evts).to_csv(p, index=False)
    return f"✅ {p}"

def do_export_cur(vf, od, fmt):
    vf = S["cur"]
    if not vf: return "❌"
    return _exp_onehot(vf, od) if fmt == "One-hot CSV (per-frame)" else _exp_boris(vf, od)

def do_export_all(od, fmt):
    if not S["done"]: return "❌"
    return "\n".join(_exp_onehot(v, od) if fmt == "One-hot CSV (per-frame)" else _exp_boris(v, od) for v in S["done"])

# ====================== Cursor JS ======================

CURSOR_JS = """
(fi, labels_json) => {
    let T=0, names=[], labels=[];
    try { const d=JSON.parse(labels_json); T=d.T; names=d.names; labels=d.labels; } catch(e) { return fi; }
    if (T===0) return fi;
    fi = Math.max(0, Math.min(Math.floor(fi), T-1));
    const cursor = document.getElementById('tl-cursor');
    if (cursor) cursor.style.left = ((fi/T)*100)+'%';
    const lbl = document.getElementById('tl-frame-label');
    if (lbl) { const cls=labels[fi]; lbl.textContent='F:'+fi+' '+(names[cls]||'?'); }
    return fi;
}
"""

# ====================== GUI ======================

GREEN_THEME = gr.themes.Soft(
    primary_hue=gr.themes.colors.green,
    secondary_hue=gr.themes.colors.emerald,
    neutral_hue=gr.themes.colors.gray,
)

with gr.Blocks(title="Animal Behavior Inference", theme=GREEN_THEME) as demo:
    gr.Markdown("# Animal Social Behavior Inference")
    gr.Markdown("HuggingFace & Local models — auto config detection — behavior filtering")

    cursor_state = gr.Textbox(value=S["_cursor_data"], visible=False)

    with gr.Row():
        with gr.Column(scale=1, min_width=260):
            gr.Markdown("### ① Select model")
            with gr.Tabs():
                with gr.TabItem("☁️ HuggingFace"):
                    repo_in = gr.Textbox(value=HF_REPO_ID, label="HF Repo ID", interactive=True)
                    hf_model_dd = gr.Dropdown(label="Model", choices=[], interactive=True)
                    hf_load_btn = gr.Button("📥 Load model", variant="primary")
                with gr.TabItem("💾 Local folder"):
                    local_dir_in = gr.Textbox(label="Model folder path", value=DEFAULT_LOCAL_MODEL_DIRS[0] if DEFAULT_LOCAL_MODEL_DIRS else "", interactive=True)
                    local_model_dd = gr.Dropdown(label="Model (.pth)", choices=[], interactive=True)
                    local_scan_btn = gr.Button("🔍 Scan folder", variant="secondary", size="sm")
                    local_load_btn = gr.Button("📥 Load model", variant="primary")
            model_st = gr.Textbox(label="Model status", interactive=False, lines=5)
            gr.Markdown("---")
            gr.Markdown("### ② Load video folder")
            vdir_in = gr.Textbox(label="Video folder path", value=DEFAULT_VIDEO_DIR)
            demo_btn = gr.Button("🎯 Load Demo", variant="secondary", size="sm")
            load_folder_btn = gr.Button("📂 Load folder", variant="secondary")
            scan_st = gr.Textbox(label="Folder status", interactive=False, lines=1)
            gr.Markdown("---")
            gr.Markdown("### ③ Inference")
            video_dd = gr.Dropdown(label="Select video", choices=[], interactive=True)
            batch_btn = gr.Button("📦 Batch inference (all videos)", variant="primary", size="lg")
            batch_log_tb = gr.Textbox(label="Batch log", interactive=False, lines=8)
            run_btn = gr.Button("🚀 Run inference (single)", variant="secondary")

        with gr.Column(scale=2, min_width=400):
            toggle_label_html = gr.HTML("<p style='color:#aaa;font-size:13px;'>Load a model to see behaviors</p>")
            behavior_toggles = gr.CheckboxGroup(label="Active behaviors (unchecked → merged to Other)", choices=[], value=[], interactive=True, visible=False)
            toggle_status = gr.Textbox(interactive=False, lines=1, visible=False, show_label=False)
            batch_prog = gr.HTML("")
            info_html = gr.HTML("<p style='color:#aaa;'>Load a model and run inference</p>")
            frame_img = gr.Image(label="Frame preview", type="numpy", interactive=False)
            timeline_html = gr.HTML("")
            scrubber = gr.Slider(minimum=0, maximum=100, step=1, value=0, label="Frame", interactive=True)
            with gr.Row():
                prev_btn = gr.Button("◀ Previous video", size="sm")
                nav_md_out = gr.Markdown("*No results yet*")
                next_btn = gr.Button("Next video ▶", size="sm")
            gr.Markdown("---")
            behavior_html = gr.HTML("<p style='color:#aaa;'>Run inference to see statistics</p>")

        with gr.Column(scale=1, min_width=260):
            gr.Markdown("### ④ Export")
            exp_fmt = gr.Dropdown(label="Output format", choices=["One-hot CSV (per-frame)", "BORIS event log"], value="One-hot CSV (per-frame)", interactive=True)
            exp_prev = gr.HTML("<p style='color:#aaa;font-size:13px;'>Run inference first</p>")
            out_dir = gr.Textbox(label="Save to", value=DEFAULT_OUTPUT_DIR)
            exp_cur = gr.Button("💾 Export current video", variant="primary")
            exp_all = gr.Button("📦 Export all (batch)")
            exp_log = gr.Textbox(label="Export log", interactive=False, lines=6)

    demo.load(list_models, [repo_in], [hf_model_dd, model_st])
    hf_load_btn.click(load_model_hf, [repo_in, hf_model_dd], [model_st, behavior_toggles, toggle_label_html])
    local_scan_btn.click(scan_local_models, [local_dir_in], [local_model_dd, model_st])
    local_load_btn.click(load_model_local, [local_dir_in, local_model_dd], [model_st, behavior_toggles, toggle_label_html])
    behavior_toggles.change(on_toggle_change, [behavior_toggles], [toggle_status])

    # Shared outputs for demo/load folder: video dropdown, status, preview frame, info, scrubber, timeline, cursor
    load_outputs = [video_dd, scan_st, frame_img, info_html, scrubber, timeline_html, cursor_state]
    demo_btn.click(load_demo_inference, [repo_in], load_outputs)
    load_folder_btn.click(scan_videos_and_preview, [vdir_in], load_outputs)

    # Video selection triggers preview (frame + scrubber setup)
    video_dd.change(on_video_select, [video_dd], [frame_img, info_html, scrubber, timeline_html, cursor_state])

    out9 = [batch_prog, info_html, frame_img, timeline_html, behavior_html, exp_prev, nav_md_out, scrubber, cursor_state]
    out10 = out9 + [batch_log_tb]

    run_btn.click(run_single, [video_dd], out9)
    batch_btn.click(run_batch, [], out10)
    scrubber.input(fn=None, inputs=[scrubber, cursor_state], outputs=[scrubber], js=CURSOR_JS)
    scrubber.change(on_scrub, inputs=[scrubber], outputs=[frame_img, info_html])
    prev_btn.click(lambda: do_nav("prev"), [], out9)
    next_btn.click(lambda: do_nav("next"), [], out9)
    exp_fmt.change(update_export_preview, [exp_fmt], [exp_prev])
    exp_cur.click(do_export_cur, [video_dd, out_dir, exp_fmt], [exp_log])
    exp_all.click(do_export_all, [out_dir, exp_fmt], [exp_log])

if __name__ == "__main__":
    demo.launch(debug=True, share=True)
