# @title gui (with BORIS label support)

# ==================== 👇 修改這裡 👇 ====================
HF_REPO_ID = "yiheng266/animal-social-models"
DEFAULT_VIDEO_DIR = "/content/drive/My Drive/traindata/kuo_validation/video(s and g)/video/train"
DEFAULT_LABEL_DIR = "/content/drive/My Drive/traindata/kuo_validation/video(s and g)/onehot"
DEFAULT_OUTPUT_DIR = "/content/drive/My Drive/trained_models/"
MAX_LABELS = 15  # pre-built dropdown slots
# ==================== 👆 修改以上即可 👆 ====================

import os, json, numpy as np, torch, torch.nn as nn, torch.optim as optim
import gradio as gr, pandas as pd, random, shutil, time, traceback
from PIL import Image, ImageFilter
from torchvision.transforms import ToTensor
from collections import Counter
from decord import VideoReader, cpu
from huggingface_hub import hf_hub_download, list_repo_files
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import f1_score, average_precision_score
from sklearn.model_selection import train_test_split
from difflib import SequenceMatcher

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ====================== Google Drive Mount Check ======================

def ensure_drive_mounted():
    if os.path.exists("/content") and not os.path.exists("/content/drive/My Drive"):
        try:
            from google.colab import drive
            drive.mount("/content/drive")
            print("✅ Google Drive mounted.")
        except Exception as e:
            print(f"⚠️ Could not mount Google Drive: {e}")
    elif os.path.exists("/content/drive/My Drive"):
        print("✅ Google Drive already mounted.")
    else:
        print("ℹ️ Not running in Colab — skipping Drive mount.")

ensure_drive_mounted()

# ====================== Models ======================

class MLPHead_CLS(nn.Module):
    def __init__(self, inf, nc, hd, dr):
        super().__init__()
        self.norm=nn.LayerNorm(inf); self.fc1=nn.Linear(inf,hd)
        self.relu=nn.ReLU(True); self.drop=nn.Dropout(dr); self.fc2=nn.Linear(hd,nc)
    def forward(self,x):
        return self.fc2(self.drop(self.relu(self.fc1(self.norm(x)))))

class MLPHead_TM(nn.Module):
    def __init__(self, inf, nc, hd, dr):
        super().__init__()
        self.norm=nn.LayerNorm(inf); self.fc1=nn.Linear(inf,hd)
        self.relu=nn.ReLU(True); self.drop=nn.Dropout(dr); self.fc2=nn.Linear(hd,nc)
    def forward(self,x):
        x=torch.mean(x,dim=1)
        return self.fc2(self.drop(self.relu(self.fc1(self.norm(x)))))

class CustomTimeSformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        from transformers import TimesformerModel
        self.backbone=TimesformerModel.from_pretrained(cfg["backbone"]["pretrained"])
        self.head=MLPHead_CLS(cfg["head"]["in_features"],cfg["num_classes"],cfg["head"]["hidden_dim"],cfg["head"]["dropout"])
    def forward(self,x):
        x=x.permute(0,2,1,3,4)
        return self.head(self.backbone(pixel_values=x).last_hidden_state[:,0])

class CustomSwin3D(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        from torchvision.models.video import swin3d_t, Swin3D_T_Weights
        self.model=swin3d_t(weights=Swin3D_T_Weights.DEFAULT)
        self.model.head=nn.Identity(); self.model.avgpool=nn.Identity()
        self.head=MLPHead_TM(cfg["head"]["in_features"],cfg["num_classes"],cfg["head"]["hidden_dim"],cfg["head"]["dropout"])
    def forward(self,x):
        x=self.model.patch_embed(x); x=self.model.pos_drop(x)
        x=self.model.features(x); x=self.model.norm(x); x=x.mean(dim=(2,3))
        return self.head(x)

def build_model(cfg):
    n=cfg["backbone"]["name"]
    if n=="TimesformerModel": return CustomTimeSformer(cfg)
    elif n=="CustomSwin3D": return CustomSwin3D(cfg)
    else: raise ValueError(f"Unknown backbone: {n}")

def rebuild_head(model, cfg, new_nc):
    hd=cfg["head"]["hidden_dim"]; dr=cfg["head"]["dropout"]; inf=cfg["head"]["in_features"]
    pool=cfg["head"].get("pool","cls_token")
    if pool=="temporal_mean": model.head=MLPHead_TM(inf,new_nc,hd,dr)
    else: model.head=MLPHead_CLS(inf,new_nc,hd,dr)
    return model

# ====================== Preprocess + Augmentation ======================

def uniform_sample(frames,t):
    n=len(frames)
    if n==t: return frames
    if n<t: return frames+[frames[-1]]*(t-n)
    return [frames[i] for i in np.linspace(0,n-1,t,dtype=int)]

def preprocess(frames,cfg):
    sz=cfg["backbone"]["input_size"]; nf=cfg["backbone"]["num_frames"]
    m=cfg["input_format"]["normalize"]["mean"]; s=cfg["input_format"]["normalize"]["std"]
    r=[f.resize((sz,sz),Image.BILINEAR) for f in frames]
    if len(r)!=nf: r=uniform_sample(r,nf)
    v=torch.stack([ToTensor()(f) for f in r],0)
    return ((v-torch.tensor(m).view(1,-1,1,1))/torch.tensor(s).view(1,-1,1,1)).permute(1,0,2,3)

def random_blur(frames,frac=0.35,rng=None):
    if frac<=0 or not frames: return frames
    rng=rng or random; n=len(frames); k=max(1,int(round(n*frac))); idxs=rng.sample(range(n),k)
    return [f.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.8,2.2))) if i in idxs else f for i,f in enumerate(frames)]

def temporal_dropout(frames,frac=0.15,rng=None):
    if frac<=0 or len(frames)<3: return frames
    rng=rng or random; n=len(frames); k=max(1,int(round(n*frac)))
    idxs=rng.sample(range(1,n-1),min(k,max(1,n-2))); out=frames[:]
    for i in idxs: out[i]=out[i-1] if rng.random()<0.5 else out[i+1]
    return out

# ====================== BORIS Label Parsing ======================
# BORIS export format: one row per START/STOP event.
# Key columns: Behavior, Behavior type (START/STOP), Time (seconds), FPS.
# Strategy: use actual video FPS from VideoReader (most accurate),
#           fall back to BORIS CSV FPS column, then finally to int(csv_fps).
# Image index column is intentionally ignored — it is often inaccurate.

def is_boris_csv(lp):
    """Return True if the CSV looks like a BORIS event export."""
    try:
        df = pd.read_csv(lp, nrows=2)
        required = {"Behavior", "Behavior type", "Time"}
        return required.issubset(set(df.columns))
    except Exception:
        return False


def boris_to_onehot(lp, total_frames, video_fps):
    """
    Convert a BORIS CSV to a per-frame one-hot (multi-hot) numpy array.

    Returns
    -------
    onehot : np.ndarray, shape (total_frames, n_behaviors + 1)
        Last column is "Other" (1 when no behaviour is active).
    behavior_names : list[str]
        Column names in order, with "Other" appended last.

    Notes
    -----
    - FPS priority: video_fps (from VideoReader) > BORIS 'FPS' column > fallback 25.
    - Frame index = int(time_seconds * fps), clipped to [0, total_frames).
    - START/STOP count mismatch for a behaviour → that behaviour is skipped with a warning.
    - Overlapping behaviours are allowed (multi-hot).
    """
    df = pd.read_csv(lp)

    # ---- resolve FPS ----
    fps = video_fps  # prefer real video fps
    if fps <= 0 and "FPS" in df.columns:
        try:
            fps_vals = pd.to_numeric(df["FPS"], errors="coerce").dropna()
            if len(fps_vals) > 0:
                fps = float(fps_vals.iloc[0])
        except Exception:
            pass
    if fps <= 0:
        fps = 25.0
        print(f"⚠️  Could not determine FPS for {lp}, defaulting to 25")

    # ---- collect unique behaviors (preserve CSV order) ----
    all_behaviors = list(dict.fromkeys(df["Behavior"].dropna().tolist()))

    n_beh = len(all_behaviors)
    onehot = np.zeros((total_frames, n_beh), dtype=np.int8)

    for bi, beh in enumerate(all_behaviors):
        bdf = df[df["Behavior"] == beh]
        starts = bdf[bdf["Behavior type"] == "START"]["Time"].values.astype(float)
        stops  = bdf[bdf["Behavior type"] == "STOP"]["Time"].values.astype(float)

        if len(starts) != len(stops):
            print(f"⚠️  {beh}: START/STOP mismatch ({len(starts)}/{len(stops)}) — skipping")
            continue

        for t_start, t_stop in zip(starts, stops):
            f_start = int(t_start * fps)
            f_stop  = min(int(t_stop  * fps), total_frames)  # inclusive end → clip
            if f_start >= total_frames:
                continue
            onehot[f_start:f_stop, bi] = 1

    # ---- build "Other" column ----
    other_col = (onehot.sum(axis=1) == 0).astype(np.int8).reshape(-1, 1)
    onehot_full = np.concatenate([onehot, other_col], axis=1)
    behavior_names = all_behaviors + ["Other"]

    return onehot_full, behavior_names


# In-memory cache: lp -> (onehot_array, behavior_names)
# Avoids re-parsing on every window during training.
_BORIS_CACHE: dict = {}


def load_label_data(lp, total_frames, video_fps):
    """
    Unified label loader.  Returns (onehot_array, behavior_names).

    - Standard one-hot CSV  → read directly with pandas.
    - BORIS event CSV       → convert via boris_to_onehot() with caching.
    """
    if is_boris_csv(lp):
        cache_key = (lp, total_frames, round(video_fps, 4))
        if cache_key not in _BORIS_CACHE:
            onehot, names = boris_to_onehot(lp, total_frames, video_fps)
            _BORIS_CACHE[cache_key] = (onehot, names)
        return _BORIS_CACHE[cache_key]
    else:
        # Original one-hot format
        df = pd.read_csv(lp)
        return df.values.astype(np.int8), list(df.columns)


# ====================== Dataset ======================

class SlidingWindowDataset(Dataset):
    def __init__(self,video_paths,label_paths,ws,stride,cfg,nc,label_map,skip=0,augment=None):
        self.cfg=cfg; self.ws=ws; self.augment=augment; self.samples=[]; self.sample_labels=[]
        for vp,lp in zip(video_paths,label_paths):
            try:
                vr=VideoReader(vp,ctx=cpu(0)); T=len(vr); fps=vr.get_avg_fps()

                # ---- unified label loading (handles both BORIS and one-hot) ----
                oh, _ = load_label_data(lp, T, fps)

                n_cols=oh.shape[1]
                if T!=len(oh): print(f"⚠️ Length mismatch {vp}"); continue
                raw=np.argmax(oh[:,:n_cols],axis=1); mapped=np.array([label_map.get(int(l),-1) for l in raw])
                sel=list(range(0,T,skip+1)); valid=[i for i in sel if i<len(mapped) and mapped[i]>=0]
                if len(valid)<ws: continue
                for s in range(0,len(valid)-ws+1,stride):
                    idx=valid[s:s+ws]; lbl=Counter(mapped[idx]).most_common(1)[0][0]
                    self.samples.append((vp,idx,int(lbl))); self.sample_labels.append(int(lbl))
            except Exception as e: print(f"⚠️ Skipped {vp}: {e}"); continue
    def __len__(self): return len(self.samples)
    def __getitem__(self,i):
        vp,idx,lbl=self.samples[i]; vr=VideoReader(vp,ctx=cpu(0))
        frames=[Image.fromarray(f) for f in vr.get_batch(idx).asnumpy()]
        if len(frames)<self.ws: frames+=[frames[-1]]*(self.ws-len(frames))
        if self.augment: frames=self.augment(frames)
        return preprocess(frames,self.cfg),torch.tensor(lbl,dtype=torch.long)

# ====================== State ======================

S = {"model":None,"cfg":None,"scan_data":None,"label_names":[],"cur_vf":None,"cur_vr":None,
     "_cursor_data":json.dumps({"T":0,"names":[],"labels":[]}),"train_log":[],"split_indices":{"train":[],"val":[]}}

CLR_PAL=["#378ADD","#D85A30","#E24B4A","#7F77DD","#1D9E75","#BA7517",
         "#534AB7","#993C1D","#639922","#D4537E","#185FA5","#854F0B","#A32D2D"]
U=gr.update()

def get_clr(i,name):
    if name.lower() in ("other","others"): return "#FFFFFF","rgba(180,180,180,0.9)"
    c=CLR_PAL[i%len(CLR_PAL)]; r,g,b=int(c[1:3],16),int(c[3:5],16),int(c[5:7],16)
    return c,f"rgba({r},{g},{b},0.9)"

# ====================== Model Management ======================

def list_models(repo):
    try:
        files=list_repo_files(repo)
        pths=[f for f in files if f.endswith("/model.pth") or f=="model.pth"]
        if not pths: pths=[f for f in files if f.endswith(".pth")]
        if not pths: return gr.update(choices=[],value=None),"❌ No models found"
        names=[os.path.dirname(p) if "/" in p else p for p in pths]
        return gr.update(choices=names,value=names[0]),f"✅ {len(names)} model(s) found"
    except Exception as e: return gr.update(choices=[],value=None),f"❌ {e}"

def load_pretrained(repo,mname):
    if not mname or not repo: return "❌ Specify repo & model"
    try:
        if not mname.endswith(".pth"): cf=f"{mname}/config.json"; pf=f"{mname}/model.pth"
        else: cf="config.json"; pf=mname
        with open(hf_hub_download(repo_id=repo,filename=cf)) as f: cfg=json.load(f)
        model=build_model(cfg)
        model.load_state_dict(torch.load(hf_hub_download(repo_id=repo,filename=pf),map_location=device,weights_only=True))
        model.to(device); S.update({"model":model,"cfg":cfg,"train_log":[]})
        return f"✅ Loaded: {mname}\n  Backbone: {cfg['backbone']['name']}\n  Classes: {cfg['class_names']}\n  Device: {device}"
    except Exception as e: traceback.print_exc(); return f"❌ {e}"

# ====================== Train/Val Split ======================

def compute_split(val_pct, seed=1337):
    if not S["scan_data"]: S["split_indices"]={"train":[],"val":[]}; return
    data=S["scan_data"]; n=len(data); val_ratio=val_pct/100.0
    if val_ratio>0 and n>=4: tidx,vidx=train_test_split(list(range(n)),test_size=val_ratio,random_state=int(seed))
    elif val_ratio>0 and n>=2: vidx=[n-1]; tidx=list(range(n-1))
    else: tidx=list(range(n)); vidx=[]
    S["split_indices"]={"train":tidx,"val":vidx}

def build_video_list_html(active_vf=None):
    if not S["scan_data"]: return "<p style='color:#aaa;font-size:12px;'>Load data first</p>"
    data=S["scan_data"]; tidx_set=set(S["split_indices"].get("train",[])); vidx_set=set(S["split_indices"].get("val",[]))
    html="<div style='max-height:200px;overflow-y:auto;border:1px solid var(--color-border-secondary);border-radius:8px;padding:4px;'>"
    for i,d in enumerate(data):
        vf=d["vf"]; T=d["T"]; fps=d["fps"]; dur=T/fps if fps>0 else 0
        fmt_tag = "<span style='font-size:10px;padding:1px 5px;border-radius:3px;background:#EDE9FE;color:#5B21B6;font-weight:600;margin-left:4px;'>BORIS</span>" if d.get("is_boris") else ""
        is_active=(vf==active_vf); is_val=(i in vidx_set)
        bg="background:rgba(220,38,38,0.12);border-left:3px solid #dc2626;" if is_active else "background:transparent;border-left:3px solid transparent;"
        if is_val: role_tag="<span style='font-size:10px;padding:1px 5px;border-radius:3px;background:#FEF3C7;color:#92400E;font-weight:600;margin-left:6px;'>VAL</span>"
        elif i in tidx_set: role_tag="<span style='font-size:10px;padding:1px 5px;border-radius:3px;background:#D1FAE5;color:#065F46;font-weight:600;margin-left:6px;'>TRAIN</span>"
        else: role_tag=""
        nc="#dc2626" if is_active else "var(--color-text-primary)"; nw="700" if is_active else "500"
        html+=f"<div style='{bg}border-radius:4px;padding:4px 8px;margin-bottom:1px;'><div style='display:flex;justify-content:space-between;align-items:center;'><span style='font-size:12px;color:{nc};font-weight:{nw};'>{vf}{fmt_tag}{role_tag}</span><span style='font-size:10px;color:var(--color-text-secondary);white-space:nowrap;'>{T} fr · {dur:.1f}s</span></div></div>"
    html+="</div>"
    nt=len(tidx_set); nv=len(vidx_set)
    html+=f"<div style='display:flex;gap:14px;margin-top:4px;font-size:11px;color:var(--color-text-secondary);'><span><span style='display:inline-block;width:8px;height:8px;border-radius:2px;background:#D1FAE5;border:1px solid #065F46;vertical-align:middle;margin-right:3px;'></span>Train: {nt}</span><span><span style='display:inline-block;width:8px;height:8px;border-radius:2px;background:#FEF3C7;border:1px solid #92400E;vertical-align:middle;margin-right:3px;'></span>Val: {nv}</span><span>Total: {len(data)}</span></div>"
    return html

# ====================== Label Mapping Logic ======================

def fuzzy_match(data_name, pretrained_names):
    """Find best fuzzy match for a data label among pretrained class names."""
    dn = data_name.lower().replace("_"," ")
    best_score, best_match = 0, None
    for pn in pretrained_names:
        pnl = pn.lower().replace("_"," ")
        if dn == pnl: return pn  # exact
        score = SequenceMatcher(None, dn, pnl).ratio()
        # Also check substring containment
        if dn in pnl or pnl in dn: score = max(score, 0.75)
        if score > best_score: best_score = score; best_match = pn
    return best_match if best_score >= 0.55 else None

def build_mapping_choices_pt(idx, data_labels, pretrained_names):
    """Pretrain head: choices = pretrained classes + Exclude (+ → Other only if no other/others in data)."""
    has_other = any(n.lower() in ("other","others") for n in data_labels)
    choices = list(pretrained_names)
    if not has_other: choices.append("→ Other")
    choices.append("→ Exclude")
    # Default: fuzzy match or pretrained others
    default = fuzzy_match(data_labels[idx], pretrained_names)
    if default is None:
        if "others" in pretrained_names: default = "others"
        elif not has_other: default = "→ Other"
        else: default = choices[0]
    return choices, default

def build_mapping_choices_new(idx, data_labels, all_mappings):
    """New head: choices = keep / merge into available / (→ Other if needed) / Exclude."""
    has_other = any(n.lower() in ("other","others") for n in data_labels)
    consumed = set()
    for i_str, val in all_mappings.items():
        i = int(i_str)
        if i == idx: continue
        if val not in (None, "", "keep", "→ Other", "→ Exclude"): consumed.add(i)
    choices = [f"{data_labels[idx]} (keep)"]
    for j, nm in enumerate(data_labels):
        if j == idx or j in consumed: continue
        choices.append(f"→ merge into {nm}")
    if not has_other: choices.append("→ Other")
    choices.append("→ Exclude")
    return choices

def parse_mapping_value(val, data_labels):
    """Parse a dropdown value to a mapping dict entry."""
    if val is None or val == "" or "(keep)" in str(val): return "keep"
    if "merge into" in str(val): return val
    if val == "→ Other": return "→ Other"
    if val == "→ Exclude": return "→ Exclude"
    return val  # pretrain mode: val is a pretrained class name

def compute_label_map_from_dropdowns(mode, dd_values, data_labels, pretrained_names):
    """Convert dropdown values → (new_class_names, label_map{old_idx: new_idx or None}).
    Excluded labels → None (frames skipped in training & evaluation).
    new_names follows CSV column order for consistency with test code."""
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
        if other_list:
            for i in other_list: mapping[i] = "others"
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

    # New head mode — follow CSV column order
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
        has_o = any(c.lower() in ("other","others") for c in new_names)
        if not has_o: new_names.append("Other")

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
            if t in new_names: label_map[i] = new_names.index(t)
            else: label_map[i] = new_names.index("Other") if "Other" in new_names else 0
        elif i in other_list:
            oidx = next((j for j,n in enumerate(new_names) if n.lower() in ("other","others")), len(new_names)-1)
            label_map[i] = oidx
        else:
            label_map[i] = 0

    return new_names, label_map

# ====================== Mapped Timeline ======================

def build_mapped_timeline(vf, mapped_names, label_map):
    """Build timeline HTML using mapped labels. None = excluded (shown as dim gray)."""
    if not S["scan_data"] or not vf: return "", S["_cursor_data"]
    d = next((x for x in S["scan_data"] if x["vf"]==vf), None)
    if not d: return "", S["_cursor_data"]

    T=d["T"]; fps=d["fps"]; raw_labels=d["labels"]
    # Map raw labels to new indices; None → -1 (excluded)
    mapped_labels = [label_map.get(l, 0) if label_map.get(l) is not None else -1 for l in raw_labels]
    names = mapped_names

    # Build bar
    segs=[]; cur,cnt=mapped_labels[0],1
    for i in range(1,T):
        if mapped_labels[i]==cur: cnt+=1
        else: segs.append((cur,cnt)); cur,cnt=mapped_labels[i],1
    segs.append((cur,cnt))

    bar=""
    for li,c in segs:
        if li == -1:
            nm="excluded"; clr="#D0D0D0"; bdr=""
        else:
            nm=names[li] if li<len(names) else "?"
            clr,_=get_clr(li,nm)
            bdr="border-top:1px solid #ccc;border-bottom:1px solid #ccc;" if clr=="#FFFFFF" else ""
        pct=(c/T)*100
        bar+=f"<div style='width:{pct:.3f}%;height:100%;background:{clr};{bdr}display:inline-block;box-sizing:border-box;opacity:{0.35 if li==-1 else 1};' title='{nm}'></div>"

    leg=""
    for i,nm in enumerate(names):
        clr,_=get_clr(i,nm); bdr="border:1px solid #ccc;" if clr=="#FFFFFF" else ""
        leg+=f"<span style='display:inline-flex;align-items:center;gap:3px;margin-right:10px;font-size:11px;color:var(--color-text-secondary);'><span style='display:inline-block;width:8px;height:8px;border-radius:2px;background:{clr};{bdr}'></span>{nm}</span>"
    # Add excluded legend if any excluded frames
    if -1 in mapped_labels:
        leg+=f"<span style='display:inline-flex;align-items:center;gap:3px;margin-right:10px;font-size:11px;color:var(--color-text-tertiary);'><span style='display:inline-block;width:8px;height:8px;border-radius:2px;background:#D0D0D0;opacity:0.5;'></span>excluded</span>"

    ml0=mapped_labels[0]
    nm0=names[ml0] if ml0>=0 and ml0<len(names) else "excluded"
    tl=f"""<div style='width:100%;padding:4px 0;'>
      <div style='position:relative;display:flex;height:16px;border-radius:4px;overflow:hidden;border:1px solid #ccc;'>
        {bar}
        <div id='tl-cursor' style='position:absolute;top:-2px;bottom:-2px;width:2px;background:#000;box-shadow:0 0 0 1px rgba(255,255,255,0.8);left:0%;pointer-events:none;'></div>
      </div>
      <div style='display:flex;justify-content:space-between;align-items:center;margin-top:3px;'>
        <div>{leg}</div>
        <span id='tl-frame-label' style='font-size:11px;font-weight:500;color:var(--color-text-secondary);'>F:0 {nm0}</span>
      </div></div>"""

    cursor_data = json.dumps({"T":T, "names":names, "labels":mapped_labels})
    S["_cursor_data"] = cursor_data
    return tl, cursor_data

def build_mapping_summary_html(mode, dd_values, data_labels, pretrained_names):
    """Build HTML summary of the mapping result."""
    new_names, label_map = compute_label_map_from_dropdowns(mode, dd_values, data_labels, pretrained_names)
    html = f"<div style='padding:6px 10px;background:var(--color-background-secondary);border-radius:8px;font-size:11px;color:var(--color-text-secondary);line-height:1.6;'>"
    html += f"<span style='font-weight:500;'>Training classes ({len(new_names)}):</span> {', '.join(new_names)}"
    html += "</div>"
    return html

# ====================== Label Distribution HTML ======================

def build_label_dist_html():
    if not S["scan_data"] or not S["label_names"]: return "<p style='color:#aaa;'>Load data to see labels</p>"
    all_label_names=S["label_names"]; matched=S["scan_data"]; total_frames=sum(d["T"] for d in matched)
    gcounts=Counter()
    for d in matched:
        for k,v in d["counts"].items(): gcounts[k]+=v
    html="<div style='padding:4px 0;'>"
    html+="<p style='font-size:13px;font-weight:500;margin:0 0 6px;'>Label distribution</p>"
    for i,nm in enumerate(all_label_names):
        c=gcounts.get(i,0); pct=100*c/max(total_frames,1); clr,_=get_clr(i,nm)
        bar_clr="#ddd" if clr=="#FFFFFF" else clr
        html+=f"<div style='margin-bottom:5px;'><div style='display:flex;align-items:center;gap:6px;margin-bottom:1px;'><span style='display:inline-block;width:8px;height:8px;border-radius:2px;background:{bar_clr};flex-shrink:0;{("border:1px solid #ccc;" if clr=="#FFFFFF" else "")}'></span><span style='font-size:12px;font-weight:500;flex:1;'>{nm}</span><span style='font-size:11px;color:#888;flex-shrink:0;'>{c:,} fr · {pct:.1f}%</span></div><div style='height:5px;background:#f0f0f0;border-radius:3px;overflow:hidden;margin-left:14px;'><div style='width:{max(pct,0.3):.1f}%;height:100%;background:{bar_clr};border-radius:3px;'></div></div></div>"
    html+="</div>"
    return html

# ====================== Data Scanning ======================

def do_scan_and_preview(vdir, ldir, val_pct, val_seed, head_mode, *dd_vals):
    N = MAX_LABELS
    empty = lambda msg: (msg,"","*Load data first*",
                         *[gr.update(visible=False,choices=[],value=None) for _ in range(N)],
                         gr.update(choices=[],value=None),
                         None,"","",gr.update(maximum=0,value=0),S["_cursor_data"],"","")

    if not vdir or not os.path.isdir(vdir): return empty(f"❌ Video dir not found")
    if not ldir or not os.path.isdir(ldir): return empty(f"❌ Label dir not found")

    vfiles=sorted([f for f in os.listdir(vdir) if f.lower().endswith((".mp4",".avi"))])
    if not vfiles: return empty("❌ No videos found")

    matched=[]; all_label_names=None
    boris_count=0; onehot_count=0

    for vf in vfiles:
        base=os.path.splitext(vf)[0]; lp=None
        for c in [base+".csv", base+"_one_hot.csv"]:
            fp=os.path.join(ldir,c)
            if os.path.exists(fp): lp=fp; break
        if lp is None: continue
        vp=os.path.join(vdir,vf)
        try:
            vr=VideoReader(vp,ctx=cpu(0)); T=len(vr); fps=vr.get_avg_fps()

            # ---- detect format and load ----
            boris = is_boris_csv(lp)
            oh, col_names = load_label_data(lp, T, fps)

            if boris:
                boris_count += 1
            else:
                onehot_count += 1

            # sanity check: one-hot CSVs must match frame count exactly
            if not boris and T != len(oh):
                print(f"⚠️ Length mismatch (one-hot) {vf}: video={T}, csv={len(oh)}")
                continue

            if all_label_names is None:
                all_label_names = col_names
            elif col_names != all_label_names:
                # Tolerate different column order / subset by re-aligning
                print(f"⚠️ Column mismatch in {vf} — will use first file's column order")

            # argmax over behaviour columns (excluding Other for counting purposes)
            n_cols = oh.shape[1]
            labels = np.argmax(oh[:, :n_cols], axis=1)
            counts = Counter(labels.tolist())
            matched.append({
                "vp": vp, "lp": lp, "vf": vf,
                "T": T, "fps": fps,
                "counts": counts,
                "labels": labels.tolist(),
                "is_boris": boris,
            })
        except Exception as e:
            print(f"⚠️ Skipped {vf}: {e}")
            continue

    if not matched: return empty("❌ No matched pairs")
    if all_label_names is None:
        all_label_names = [f"class_{i}" for i in range(max(max(d["counts"].keys()) for d in matched)+1)]

    S["scan_data"]=matched; S["label_names"]=all_label_names
    compute_split(val_pct, val_seed)
    dist=build_label_dist_html()

    # format summary for status bar
    fmt_parts = []
    if boris_count:   fmt_parts.append(f"{boris_count} BORIS")
    if onehot_count:  fmt_parts.append(f"{onehot_count} one-hot")
    fmt_str = f" ({', '.join(fmt_parts)})" if fmt_parts else ""

    pretrained_names = S["cfg"]["class_names"] if S["cfg"] else []

    # Build mapping dropdown updates
    dd_updates = []
    for i in range(N):
        if i < len(all_label_names):
            if head_mode == "Pretrain head" and pretrained_names:
                choices, default = build_mapping_choices_pt(i, all_label_names, pretrained_names)
            else:
                choices = build_mapping_choices_new(i, all_label_names, {})
                default = choices[0]
            dd_updates.append(gr.update(visible=True, choices=choices, value=default, label=all_label_names[i]))
        else:
            dd_updates.append(gr.update(visible=False, choices=[], value=None))

    # Video dropdown
    vnames=[d["vf"] for d in matched]
    vid_dd_update=gr.update(choices=vnames,value=vnames[0])

    # Preview first video with initial mapping
    vf=matched[0]["vf"]
    dd_values = [u["value"] for u in dd_updates[:len(all_label_names)]]
    new_names, label_map = compute_label_map_from_dropdowns(head_mode, dd_values, all_label_names, pretrained_names)
    tl, cdata = build_mapped_timeline(vf, new_names, label_map)
    summary = build_mapping_summary_html(head_mode, dd_values, all_label_names, pretrained_names)

    img = _get_frame(vf, 0)
    d0 = next(x for x in matched if x["vf"]==vf)
    T=d0["T"]; fps=d0["fps"]
    ml = label_map.get(d0["labels"][0], 0)
    nm0 = new_names[ml] if ml is not None and ml < len(new_names) else "?"
    _,bg = get_clr(ml if ml is not None else 0, nm0)
    info = f"<div style='display:flex;justify-content:space-between;align-items:center;'><span style='padding:3px 10px;border-radius:6px;background:{bg};color:white;font-size:12px;font-weight:500;'>{nm0}</span><span style='font-size:12px;color:var(--color-text-secondary);'>F: 0 / {T} | 0.00s / {T/fps:.2f}s</span></div>"

    nav_t=f"**{vf}** — 1 / {len(matched)} videos"
    vid_list=build_video_list_html(active_vf=vf)
    status=f"✅ {len(matched)} matched (of {len(vfiles)} videos){fmt_str}"

    return (status, dist, nav_t, *dd_updates, vid_dd_update,
            img, info, tl, gr.update(maximum=max(T-1,0),value=0), cdata, vid_list, summary)


# ====================== Mapping Change Handler ======================

def on_mapping_change(head_mode, *dd_vals):
    """When any mapping dropdown or head mode changes → rebuild all dropdowns + timeline + summary."""
    data_labels = S["label_names"]
    pretrained_names = S["cfg"]["class_names"] if S["cfg"] else []
    N = MAX_LABELS
    n = len(data_labels)

    if n == 0:
        # must match map_change_outputs: [*map_dds, timeline_html, cursor_state, mapping_summary]
        return (*[gr.update() for _ in range(N)], "", S["_cursor_data"], "")

    # Parse current values
    cur_vals = list(dd_vals[:N])

    if head_mode == "Pretrain head":
        # Pretrain: choices are static, no need to rebuild
        dd_updates = []
        for i in range(N):
            if i < n:
                choices, default = build_mapping_choices_pt(i, data_labels, pretrained_names)
                current = cur_vals[i]
                if current in choices:
                    dd_updates.append(gr.update(choices=choices, value=current, label=data_labels[i]))
                else:
                    dd_updates.append(gr.update(choices=choices, value=default, label=data_labels[i]))
            else:
                dd_updates.append(gr.update())
    else:
        # New head: rebuild choices dynamically (consumed labels disappear)
        mappings = {}
        for i in range(n):
            v = cur_vals[i] if i < len(cur_vals) else "keep"
            mappings[str(i)] = parse_mapping_value(v, data_labels)

        dd_updates = []
        for i in range(N):
            if i < n:
                choices = build_mapping_choices_new(i, data_labels, mappings)
                current = cur_vals[i]
                if current in choices:
                    dd_updates.append(gr.update(choices=choices, value=current))
                else:
                    dd_updates.append(gr.update(choices=choices, value=choices[0]))
            else:
                dd_updates.append(gr.update())

    # Compute final mapping + rebuild timeline
    final_vals = [dd_updates[i].get("value", cur_vals[i]) if isinstance(dd_updates[i], dict) and "value" in dd_updates[i] else cur_vals[i] for i in range(n)]
    new_names, label_map = compute_label_map_from_dropdowns(head_mode, final_vals, data_labels, pretrained_names)

    vf = S["cur_vf"]
    tl, cdata = build_mapped_timeline(vf, new_names, label_map) if vf else ("", S["_cursor_data"])
    summary = build_mapping_summary_html(head_mode, final_vals, data_labels, pretrained_names)

    return (*dd_updates, tl, cdata, summary)

def on_head_mode_change(head_mode, *dd_vals):
    """When head mode toggles, rebuild all dropdown choices for the new mode."""
    return on_mapping_change(head_mode, *dd_vals)

# ====================== Video Preview ======================

def _get_frame(vf, fi):
    if not S["scan_data"]: return None
    d=next((x for x in S["scan_data"] if x["vf"]==vf),None)
    if not d: return None
    try:
        if S["cur_vf"]!=vf or S["cur_vr"] is None:
            S["cur_vr"]=VideoReader(d["vp"],ctx=cpu(0)); S["cur_vf"]=vf
        T=len(S["cur_vr"]); fi=max(0,min(int(fi),T-1))
        return S["cur_vr"][fi].asnumpy()
    except: S["cur_vr"]=None; S["cur_vf"]=None; return None

def _preview_video_mapped(vf, head_mode, dd_vals):
    """Preview video with current mapping applied."""
    if not S["scan_data"] or not vf: return None,"","",U,S["_cursor_data"]
    d=next((x for x in S["scan_data"] if x["vf"]==vf),None)
    if not d: return None,"","",U,S["_cursor_data"]

    data_labels=S["label_names"]; pretrained_names=S["cfg"]["class_names"] if S["cfg"] else []
    new_names, label_map = compute_label_map_from_dropdowns(head_mode, list(dd_vals[:len(data_labels)]), data_labels, pretrained_names)

    T=d["T"]; fps=d["fps"]
    tl, cdata = build_mapped_timeline(vf, new_names, label_map)

    ml = label_map.get(d["labels"][0], 0)
    nm0 = new_names[ml] if ml is not None and ml < len(new_names) else "?"
    _,bg = get_clr(ml if ml is not None else 0, nm0)
    info = f"<div style='display:flex;justify-content:space-between;align-items:center;'><span style='padding:3px 10px;border-radius:6px;background:{bg};color:white;font-size:12px;font-weight:500;'>{nm0}</span><span style='font-size:12px;color:var(--color-text-secondary);'>F: 0 / {T} | 0.00s / {T/fps:.2f}s</span></div>"

    img = _get_frame(vf, 0)
    return img, info, tl, gr.update(maximum=max(T-1,0),value=0), cdata

def on_scrub(fi, head_mode, *dd_vals):
    vf=S["cur_vf"]
    if not vf or not S["scan_data"]: return None,"<p style='color:#aaa;'>No data</p>"
    d=next((x for x in S["scan_data"] if x["vf"]==vf),None)
    if not d: return None,""

    data_labels=S["label_names"]; pretrained_names=S["cfg"]["class_names"] if S["cfg"] else []
    new_names,label_map=compute_label_map_from_dropdowns(head_mode,list(dd_vals[:len(data_labels)]),data_labels,pretrained_names)

    T=d["T"]; fps=d["fps"]; fi=max(0,min(int(fi),T-1))
    ml=label_map.get(d["labels"][fi],0)
    nm=new_names[ml] if ml is not None and ml<len(new_names) else "?"
    _,bg=get_clr(ml if ml is not None else 0,nm)
    info=f"<div style='display:flex;justify-content:space-between;align-items:center;'><span style='padding:3px 10px;border-radius:6px;background:{bg};color:white;font-size:12px;font-weight:500;'>{nm}</span><span style='font-size:12px;color:var(--color-text-secondary);'>F: {fi} / {T} | {fi/fps:.2f}s / {T/fps:.2f}s</span></div>"
    return _get_frame(vf,fi), info

def do_nav(direction, head_mode, *dd_vals):
    if not S["scan_data"]: return None,"","",U,S["_cursor_data"],"*No data*",""
    vnames=[d["vf"] for d in S["scan_data"]]; cur=S["cur_vf"]
    idx=vnames.index(cur) if cur in vnames else 0
    if direction=="prev": idx=max(0,idx-1)
    else: idx=min(len(vnames)-1,idx+1)
    vf=vnames[idx]
    img,info,tl,scrub,cdata=_preview_video_mapped(vf,head_mode,dd_vals)
    vid_list=build_video_list_html(active_vf=vf)
    return img,info,tl,scrub,cdata,f"**{vf}** — {idx+1} / {len(vnames)} videos",vid_list

def on_vid_change(vf, head_mode, *dd_vals):
    img,info,tl,scrub,cdata=_preview_video_mapped(vf,head_mode,dd_vals)
    vid_list=build_video_list_html(active_vf=vf)
    idx=0; total=0
    if S["scan_data"]:
        vnames=[d["vf"] for d in S["scan_data"]]
        total=len(vnames)
        if vf in vnames: idx=vnames.index(vf)
    nav_txt=f"**{vf}** — {idx+1} / {total} videos" if total else "*Load data first*"
    return img,info,tl,scrub,cdata,vid_list,nav_txt

def on_val_ratio_change(val_pct, val_seed):
    compute_split(val_pct, val_seed)
    return build_video_list_html(active_vf=S["cur_vf"])

# ====================== Progress + Validation HTML ======================

def html_progress(ep_done,ep_total,win_done,win_total,phase="training"):
    if ep_total==0: return ""
    ep_pct=(ep_done/ep_total)*100; wp=(win_done/max(win_total,1))*100
    ec="#1D9E75" if ep_done==ep_total else "#D85A30"
    st="✅ Complete" if ep_done==ep_total else "Training..."
    return f"<div style='background:#fff;border:1px solid #e0e0e0;border-radius:8px;padding:10px 14px;'><div style='display:flex;justify-content:space-between;margin-bottom:4px;'><span style='font-size:13px;font-weight:500;'>Epoch — {st}</span><span style='font-size:12px;color:#888;'>{ep_done}/{ep_total} epochs</span></div><div style='height:8px;background:#eee;border-radius:4px;overflow:hidden;margin-bottom:10px;'><div style='width:{ep_pct:.1f}%;height:100%;background:{ec};border-radius:4px;transition:width 0.3s;'></div></div><div style='display:flex;justify-content:space-between;margin-bottom:4px;'><span style='font-size:12px;font-weight:500;'>Epoch {min(ep_done+1,ep_total)} — {phase}</span><span style='font-size:12px;color:#888;'>{win_done}/{win_total} windows</span></div><div style='height:6px;background:#eee;border-radius:3px;overflow:hidden;'><div style='width:{wp:.1f}%;height:100%;background:#1D9E75;border-radius:3px;transition:width 0.15s;'></div></div></div>"

def html_val_card(epoch,loss,f1,mAP,f1_per,ap_per,names,is_best=False):
    brd="border:2px solid var(--color-border-info);" if is_best else "border:0.5px solid var(--color-border-tertiary);"
    badge="<span style='font-size:10px;padding:2px 6px;background:var(--color-background-info);color:var(--color-text-info);border-radius:var(--border-radius-md);margin-left:6px;'>best</span>" if is_best else ""
    fc="color:var(--color-text-info);" if is_best else ""
    rows="".join(f"<div style='display:flex;justify-content:space-between;'><span>{nm}</span><span>F1: {f1_per[i] if i<len(f1_per) else 0:.3f} · AP: {ap_per[i] if i<len(ap_per) else 0:.3f}</span></div>" for i,nm in enumerate(names))
    return f"<div style='background:var(--color-background-primary);{brd}border-radius:var(--border-radius-lg);padding:14px;margin-bottom:12px;'><div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;'><div style='display:flex;align-items:center;'><span style='font-size:13px;font-weight:500;'>Epoch {epoch}</span>{badge}</div><span style='font-size:11px;color:var(--color-text-secondary);'>loss: {loss:.4f}</span></div><div style='display:flex;gap:8px;margin-bottom:8px;'><div style='flex:1;background:var(--color-background-secondary);border-radius:var(--border-radius-md);padding:6px;text-align:center;'><div style='font-size:11px;color:var(--color-text-secondary);'>F1-macro</div><div style='font-size:16px;font-weight:500;{fc}'>{f1:.4f}</div></div><div style='flex:1;background:var(--color-background-secondary);border-radius:var(--border-radius-md);padding:6px;text-align:center;'><div style='font-size:11px;color:var(--color-text-secondary);'>mAP</div><div style='font-size:16px;font-weight:500;{fc}'>{mAP:.4f}</div></div></div><div style='font-size:11px;color:var(--color-text-secondary);line-height:1.6;'>{rows}</div></div>"

def build_val_html(log,names):
    if not log: return "<p style='color:#aaa;'>Training not started</p>"
    best=max(range(len(log)),key=lambda i:log[i]["f1"])
    return "".join(html_val_card(e["epoch"],e["loss"],e["f1"],e["mAP"],e["f1_per"],e["ap_per"],names,i==best) for i,e in enumerate(log))

# ====================== Training ======================

def run_training(repo,mname,vdir,ldir,odir,head_mode,
                 n_epochs,batch_sz,lr_str,val_pct,ws,stride_val,val_seed,train_seed,*dd_vals):
    try:
        lr=float(lr_str); val_ratio=float(val_pct)/100.0; n_epochs=int(n_epochs)
        batch_sz=int(batch_sz); ws=int(ws); stride_val=int(stride_val)
        val_seed=int(val_seed); train_seed=int(train_seed)
    except Exception as e: yield f"❌ Invalid params: {e}",U; return

    if not S["model"] or not S["cfg"]: yield "❌ Load model first",U; return
    if not S["scan_data"]: yield "❌ Load data first",U; return

    cfg=S["cfg"]; data_labels=S["label_names"]
    pretrained_names=cfg["class_names"]
    os.makedirs(odir,exist_ok=True)

    # Compute label map from dropdown values
    vals = list(dd_vals[:len(data_labels)])
    new_names, label_map = compute_label_map_from_dropdowns(head_mode, vals, data_labels, pretrained_names)
    new_nc = len(new_names)

    print(f"🏷️ Label mapping: {label_map}")
    print(f"🏷️ Training classes ({new_nc}): {new_names}")

    # Deep copy pretrained model so S["model"] stays pristine for re-training
    import copy
    model = copy.deepcopy(S["model"])
    model = rebuild_head(model, cfg, new_nc).to(device)

    data=S["scan_data"]; vps=[d["vp"] for d in data]; lps=[d["lp"] for d in data]
    tidx=S["split_indices"]["train"]; vidx=S["split_indices"]["val"]
    if not tidx and not vidx:
        if val_ratio>0 and len(data)>=4: tidx,vidx=train_test_split(list(range(len(data))),test_size=val_ratio,random_state=val_seed)
        else: tidx=list(range(len(data))); vidx=[]

    yield html_progress(0,n_epochs,0,0,"building dataset..."),"<p style='color:#aaa;'>Building...</p>"

    aug_rng=random.Random(train_seed)
    def aug(fr): return temporal_dropout(random_blur(fr,rng=aug_rng),rng=aug_rng)

    train_ds=SlidingWindowDataset([vps[i] for i in tidx],[lps[i] for i in tidx],ws,stride_val,cfg,new_nc,label_map,augment=aug)
    val_ds=SlidingWindowDataset([vps[i] for i in vidx],[lps[i] for i in vidx],ws,stride_val,cfg,new_nc,label_map) if vidx else None

    if len(train_ds)==0: yield "❌ No training windows created.",""; return

    train_loader=DataLoader(train_ds,batch_sz,shuffle=True,num_workers=4,pin_memory=True)
    val_loader=DataLoader(val_ds,batch_sz,shuffle=False,num_workers=4,pin_memory=True) if val_ds and len(val_ds)>0 else None
    total_win=len(train_ds)

    optimizer=optim.AdamW(model.parameters(),lr=lr,weight_decay=0.01)
    scheduler=CosineAnnealingLR(optimizer,T_max=n_epochs)
    scaler=GradScaler(); criterion=nn.CrossEntropyLoss(); accum=2
    S["train_log"]=[]

    for ep in range(n_epochs):
        model.train(); rl=0.0; optimizer.zero_grad(set_to_none=True); nb=len(train_loader)
        for bi,(vids,tgts) in enumerate(train_loader):
            vids=vids.to(device); tgts=tgts.to(device)
            with autocast(): loss=criterion(model(vids),tgts)/accum
            scaler.scale(loss).backward()
            if (bi+1)%accum==0 or (bi+1)==nb: scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
            rl+=loss.item()*accum*vids.size(0)
            if (bi+1)%5==0 or bi==nb-1: yield html_progress(ep,n_epochs,min((bi+1)*batch_sz,total_win),total_win,"training"),U

        scheduler.step(); ep_loss=rl/len(train_ds)
        f1m=0; mAP=0; f1p=[]; app=[]
        if val_loader:
            model.eval(); ap_=[]; al_=[]; apr_=[]; val_total=len(val_ds); nb_val=len(val_loader)
            with torch.no_grad():
                for vi,(v,t) in enumerate(val_loader):
                    v=v.to(device)
                    with autocast(): o=model(v); pr=torch.softmax(o,dim=1)
                    ap_.extend(torch.argmax(o,1).cpu().numpy()); al_.extend(t.numpy()); apr_.extend(pr.cpu().numpy())
                    if (vi+1)%5==0 or vi==nb_val-1:
                        vd=min((vi+1)*batch_sz,val_total)
                        yield html_progress(ep,n_epochs,vd,val_total,"validating"),U
            f1p=f1_score(al_,ap_,average=None,labels=list(range(new_nc)),zero_division=0).tolist()
            f1m=f1_score(al_,ap_,average="macro",zero_division=0)
            oh=np.zeros((len(al_),new_nc))
            for i,l in enumerate(al_): oh[i,l]=1
            pr_arr=np.array(apr_)
            for ci in range(new_nc):
                try: app.append(average_precision_score(oh[:,ci],pr_arr[:,ci]))
                except: app.append(0.0)
            mAP=np.mean(app)

        mp=os.path.join(odir,f"epoch_{ep+1}_f1_{f1m:.4f}_map_{mAP:.4f}.pth")
        torch.save(model.state_dict(),mp)

        # Save config.json alongside .pth — compatible with test code
        cfg_out = {
            "model_info": {
                "backbone": cfg["backbone"]["name"],
                "head": {"in_features": cfg["head"]["in_features"], "hidden_dim": cfg["head"]["hidden_dim"],
                         "dropout": cfg["head"]["dropout"], "pool": cfg["head"].get("pool","cls_token")},
                "input_format": cfg.get("input_format", {}),
                "backbone_config": {"input_size": cfg["backbone"].get("input_size",224),
                                    "num_frames": cfg["backbone"].get("num_frames",8)},
            },
            # Fields the test code needs directly:
            "ALL_BEHAVIOR_NAMES": list(data_labels),
            "SELECTED_BEHAVIORS": list(new_names),
            "num_classes": new_nc,
            "class_names": list(new_names),
            # original_to_new: same format as test code
            # selected → sequential index, unselected → null (test code treats as None → exclude)
            "original_to_new": {
                str(i): label_map[i] if i in label_map else None
                for i in range(len(data_labels))
            },
            # Full mapping details for reference
            "mapping_mode": head_mode,
            "mapping_detail": {
                data_labels[i]: {"target": new_names[label_map[i]] if i in label_map and label_map[i] is not None else "excluded",
                                 "target_idx": label_map.get(i, None)}
                for i in range(len(data_labels))
            },
            "training_params": {
                "epochs": n_epochs, "batch_size": batch_sz, "lr": lr,
                "window_size": ws, "stride": stride_val,
                "val_seed": val_seed, "train_seed": train_seed,
            },
        }
        cfg_path = mp.replace(".pth", "_config.json")
        with open(cfg_path, "w") as f: json.dump(cfg_out, f, indent=2)

        S["train_log"].append({"epoch":ep+1,"loss":ep_loss,"f1":f1m,"mAP":mAP,"f1_per":f1p,"ap_per":app,"path":mp,"config_path":cfg_path})
        yield html_progress(ep+1,n_epochs,total_win,total_win,"done"),build_val_html(S["train_log"],new_names)

    with open(os.path.join(odir,"training_log.json"),"w") as f: json.dump(S["train_log"],f,indent=2)
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

YELLOW_THEME = gr.themes.Soft(primary_hue=gr.themes.colors.amber, secondary_hue=gr.themes.colors.yellow, neutral_hue=gr.themes.colors.gray)

# ====================== Demo Loading ======================

DEMO_LOCAL_DIR = os.path.join(os.path.expanduser("~"), "demo_data")

def load_demo_training(repo):
    """Download ALL files from HF repo demo/ folder into a local dir.
    Sets both video dir and label dir to the same folder (scan matches by filename)."""
    if not repo:
        return "❌ Specify repo", "", ""
    try:
        all_files = list_repo_files(repo)
        demo_files = [f for f in all_files if f.startswith("demo/") and f != "demo/"]
        if not demo_files:
            return "❌ No files in demo/ folder on HuggingFace", "", ""
        os.makedirs(DEMO_LOCAL_DIR, exist_ok=True)
        downloaded = []
        for f in demo_files:
            local = hf_hub_download(repo_id=repo, filename=f)
            fname = os.path.basename(f)
            dest = os.path.join(DEMO_LOCAL_DIR, fname)
            if not os.path.exists(dest) or os.path.getsize(dest) != os.path.getsize(local):
                shutil.copy2(local, dest)
            downloaded.append(fname)
        n_vid = len([f for f in downloaded if f.lower().endswith((".mp4", ".avi", ".mov"))])
        n_csv = len([f for f in downloaded if f.lower().endswith(".csv")])
        return (f"✅ Demo loaded: {len(downloaded)} file(s) ({n_vid} video, {n_csv} csv)",
                DEMO_LOCAL_DIR, DEMO_LOCAL_DIR)
    except Exception as e:
        return f"❌ {e}", "", ""

# ====================== GUI ======================

with gr.Blocks(title="Training", theme=YELLOW_THEME) as demo:
    gr.Markdown("# 🏋️ Animal Behavior Model Training")
    gr.Markdown("Fine-tune from pretrained — preview labels & configure mapping before training")

    cursor_state = gr.Textbox(value=S["_cursor_data"], visible=False)
    repo_in = gr.Textbox(value=HF_REPO_ID, visible=False)

    with gr.Row():
        # ===== LEFT =====
        with gr.Column(scale=1, min_width=250):
            gr.Markdown("### ① Select model")
            model_dd=gr.Dropdown(label="Base model",choices=[],interactive=True)
            load_btn=gr.Button("📥 Load pretrained",variant="primary")
            model_st=gr.Textbox(label="Status",interactive=False,lines=4)
            gr.Markdown("---")
            gr.Markdown("### ② Load data")
            vdir_in=gr.Textbox(label="Video directory",value=DEFAULT_VIDEO_DIR)
            ldir_in=gr.Textbox(label="Label directory",value=DEFAULT_LABEL_DIR)
            odir_in=gr.Textbox(label="Output directory",value=DEFAULT_OUTPUT_DIR)
            demo_btn=gr.Button("🎯 Load Demo",variant="secondary",size="sm")
            scan_d=gr.Button("📂 Load folder",variant="secondary")
            scan_st=gr.Textbox(label="Folder status",interactive=False,lines=1)

        # ===== CENTER =====
        with gr.Column(scale=2, min_width=400):
            progress_html=gr.HTML("")
            info_html=gr.HTML("<p style='color:#aaa;'>Load data to preview</p>")
            frame_img=gr.Image(label="Frame preview",type="numpy",interactive=False)
            timeline_html=gr.HTML("")
            scrubber=gr.Slider(minimum=0,maximum=100,step=1,value=0,label="Frame",interactive=True)
            with gr.Row():
                prev_btn=gr.Button("◀ Prev",size="sm")
                nav_md=gr.Markdown("*Load data first*")
                next_btn=gr.Button("Next ▶",size="sm")
            gr.Markdown("---")
            with gr.Accordion("📹 Videos", open=False):
                vid_list_html=gr.HTML("<p style='color:#aaa;font-size:12px;'>Load data first</p>")
            gr.Markdown("---")

            # Label distribution
            with gr.Accordion("📊 Label distribution", open=False):
                label_dist_html=gr.HTML("<p style='color:#aaa;'>Load data to see labels</p>")

            # Head type + mapping dropdowns
            gr.Markdown("### 🏷️ Label mapping")
            head_mode_dd=gr.Dropdown(label="Head type",choices=["Pretrain head","New head"],value="Pretrain head",interactive=True)

            # Pre-build MAX_LABELS dropdown slots (hidden by default)
            map_dds = []
            for i in range(MAX_LABELS):
                dd = gr.Dropdown(label=f"label_{i}", choices=[], value=None, interactive=True, visible=False)
                map_dds.append(dd)

            mapping_summary=gr.HTML("")

            # Hidden video dropdown
            vid_dd=gr.Dropdown(label="Video",choices=[],interactive=True,visible=False)

        # ===== RIGHT =====
        with gr.Column(scale=1, min_width=280):
            gr.Markdown("### ③ Train")
            vr_in=gr.Slider(minimum=0,maximum=50,step=5,value=15,label="Validation ratio (%)",interactive=True)
            with gr.Row():
                ep_in=gr.Number(label="Epochs",value=5,precision=0)
                bs_in=gr.Number(label="Batch",value=8,precision=0)
            with gr.Row():
                lr_in=gr.Textbox(label="LR",value="3.8e-5")
            with gr.Row():
                ws_in=gr.Number(label="Window",value=16,precision=0)
                st_in=gr.Number(label="Stride",value=4,precision=0)
            with gr.Row():
                val_seed_in=gr.Number(label="Val seed",value=1337,precision=0,info="Split reproducibility")
                train_seed_in=gr.Number(label="Train seed",value=2025,precision=0,info="Augmentation reproducibility")
            train_btn=gr.Button("🚀 Start training",variant="primary",size="lg")
            gr.Markdown("---")
            gr.Markdown("### ④ Validation results")
            val_html=gr.HTML("<p style='color:#aaa;'>Training not started</p>")

    # ===== WIRING =====

    demo.load(list_models,[repo_in],[model_dd,model_st])
    load_btn.click(load_pretrained,[repo_in,model_dd],[model_st])

    # Demo button → download from HF and fill paths
    demo_btn.click(load_demo_training,[repo_in],[scan_st,vdir_in,ldir_in])

    # Scan outputs: status, dist, nav, N dropdown updates, vid_dd, img, info, tl, scrubber, cursor, vid_list, summary
    scan_outputs = [scan_st, label_dist_html, nav_md, *map_dds, vid_dd,
                    frame_img, info_html, timeline_html, scrubber, cursor_state, vid_list_html, mapping_summary]
    scan_d.click(do_scan_and_preview, [vdir_in, ldir_in, vr_in, val_seed_in, head_mode_dd, *map_dds], scan_outputs)

    # Head mode change → rebuild all mapping dropdowns + timeline + summary
    map_change_outputs = [*map_dds, timeline_html, cursor_state, mapping_summary]
    head_mode_dd.change(on_head_mode_change, [head_mode_dd, *map_dds], map_change_outputs)

    # Any mapping dropdown change → rebuild others + timeline + summary
    for dd in map_dds:
        dd.change(on_mapping_change, [head_mode_dd, *map_dds], map_change_outputs)

    # Val ratio or val seed change → recompute split
    vr_in.change(on_val_ratio_change,[vr_in,val_seed_in],[vid_list_html])
    val_seed_in.change(on_val_ratio_change,[vr_in,val_seed_in],[vid_list_html])

    # Video dropdown → preview with mapping
    vid_dd.change(on_vid_change,[vid_dd, head_mode_dd, *map_dds],
                  [frame_img,info_html,timeline_html,scrubber,cursor_state,vid_list_html,nav_md])

    # Scrubber
    scrubber.input(fn=None,inputs=[scrubber,cursor_state],outputs=[scrubber],js=CURSOR_JS)
    scrubber.change(on_scrub,[scrubber, head_mode_dd, *map_dds],[frame_img,info_html])

    # Nav
    prev_btn.click(lambda hm,*dd: do_nav("prev",hm,*dd),[head_mode_dd,*map_dds],
                   [frame_img,info_html,timeline_html,scrubber,cursor_state,nav_md,vid_list_html])
    next_btn.click(lambda hm,*dd: do_nav("next",hm,*dd),[head_mode_dd,*map_dds],
                   [frame_img,info_html,timeline_html,scrubber,cursor_state,nav_md,vid_list_html])

    # Training — pass head_mode + all mapping dropdowns instead of label_cb
    train_btn.click(run_training,
                    [repo_in,model_dd,vdir_in,ldir_in,odir_in,head_mode_dd,
                     ep_in,bs_in,lr_in,vr_in,ws_in,st_in,val_seed_in,train_seed_in,*map_dds],
                    [progress_html,val_html])

demo.launch(debug=True,share=True)
