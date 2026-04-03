# Animal Social Behavior Inference

Video-based animal behavior classification using TimeSformer and Swin3D backbones.  
Pretrained models are available on HuggingFace: [yiheng266/animal-social-models](https://huggingface.co/yiheng266/animal-social-models)

---

## Notebooks

| Notebook | Description | |
|----------|-------------|---|
| `inference.ipynb` | Load a model and run behavior inference on your videos | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yiheng266/open-animal-behavior/blob/main/inference.ipynb) |
| `training.ipynb` | Fine-tune a pretrained model on your own labeled videos | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yiheng266/open-animal-behavior/blob/main/training.ipynb) |

---

## Data Availability

The video dataset used in this paper is not publicly available due to privacy constraints.  
However, the code and models can be applied to your own videos in the same format.

---

## Repository Structure

```
├── inference.ipynb     # GUI for running inference on your videos
├── training.ipynb      # GUI for fine-tuning on your labeled videos
├── models.py           # CustomTimeSformer, CustomSwin3D
├── config_utils.py     # Config compatibility layer (multi-format support)
├── inference.py        # Sliding-window inference core
└── requirements.txt
```

---

## Quick Start (Colab)

**Inference** — click the badge above to:
1. Install dependencies
2. Load a model from HuggingFace
3. Run inference on your videos
4. Export results as CSV or BORIS event log

**Training** — click the training badge to:
1. Load a pretrained model as backbone
2. Configure label mapping for your dataset
3. Fine-tune with sliding-window training
4. Evaluate with frame-wise F1 and mAP

---

## Programmatic Usage

```python
import torch
from config_utils import normalize_config
from models import build_model_from_config
from inference import infer_video_gen
import json

# Load config & model
with open("config.json") as f:
    cfg, _ = normalize_config(json.load(f))

model = build_model_from_config(cfg)
model.load_state_dict(torch.load("model.pth", map_location="cpu"))
model.eval()

# Run inference
for msg in infer_video_gen("videos/", "example.mp4", model, cfg):
    if isinstance(msg, dict):
        result = msg   # final result
    else:
        print(f"Progress: {msg[0]}/{msg[1]} windows")

print(result["frame_labels"])
```

---

## Models

| Model | Backbone | Classes |
|-------|----------|---------|
| Fly copulation | TimeSformer | 3 |
| Fly aggression | Swin3D | 4 |

Download from HuggingFace:
```python
from huggingface_hub import hf_hub_download
pth = hf_hub_download("yiheng266/animal-social-models", "your_model/model.pth")
```
