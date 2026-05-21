# Mouse Behavior Classification — Video Swin-T vs TimeSformer

3-fold cross-validation pipeline for classifying mouse social behaviors from video, comparing two Kinetics-400 pretrained backbones:

| Model | Backbone | Params | Pretrained On | Source |
|---|---|---|---|---|
| **Video Swin-T** | Swin Transformer-Tiny | ~28 M | Kinetics-400 | torchvision |
| **TimeSformer** | ViT-Base | ~86 M | Kinetics-400 | HuggingFace (`facebook/timesformer-base-finetuned-k400`) |

Both models use an identical MLP classification head for fair comparison.

## Behavior Classes

Default selected (5 classes):

`Aggression` · `Investigation` · `Allo-groom` · `Standing` · `Other`

Two additional classes (`Self-groom`, `Chasing`) are defined but excluded by default. Modify `--selected_behaviors` to include them.


## Data & Trained Models

Videos and pre-trained model checkpoints are available **upon request**.

Each label CSV has 7 columns (one per behavior in `ALL_BEHAVIOR_NAMES`) with one-hot encoding, one row per video frame.

## 3-Fold Cross-Validation

Videos are pre-assigned into folders `1/`, `2/`, `3/`. Each fold uses one folder for testing and the other two for training:

| Fold | `--test_folds` | `--train_folds` |
|------|----------------|-----------------|
| Fold 1 | `1` | `3 2` |
| Fold 2 | `2` | `1 3` |
| Fold 3 | `3` | `1 2` |

## Setup

```bash
pip install -r requirements.txt
```

## Training

### Video Swin-T

```bash
# Fold 1 (default)
python train_swin3d.py

# Fold 2
python train_swin3d.py --test_folds 2 --train_folds 1 3

# Fold 3
python train_swin3d.py --test_folds 3 --train_folds 1 2

# Custom paths
python train_swin3d.py --base_video_dir /path/to/videos --label_dir /path/to/labels
```

### TimeSformer (ViT-Base)

```bash
# Fold 1 (default)
python train_timesformer.py

# Fold 2
python train_timesformer.py --test_folds 2 --train_folds 1 3

# Fold 3
python train_timesformer.py --test_folds 3 --train_folds 1 2

# Data efficiency analysis (e.g. train on 60% of data)

```

### Key Training Arguments

| Argument | Default (Swin) | Default (TimeSformer) | Description |
|---|---|---|---|
| `--base_lr` | `3.8e-5` | `3e-5` | Learning rate |
| `--num_epochs` | `5` | `5` | Training epochs |
| `--batch_size` | `8` | `8` | Batch size |
| `--accumulation_steps` | `2` | `2` | Gradient accumulation |
| `--window_size` | `16` | `16` | Sliding window size (frames) |
| `--stride` | `4` | `4` | Window stride |
| `--mlp_hidden_dim` | `512` | `512` | MLP head hidden dim |
| `--mlp_dropout` | `0.3` | `0.3` | MLP head dropout |
| `--model_save_dir` | `checkpoints/swin3d` | `checkpoints/timesformer` | Output directory |
| `--use_class_weights` | off | off | Enable inverse-frequency weighting |
| `--hf_model` | — | `facebook/timesformer-base-finetuned-k400` | HuggingFace model name (TimeSformer only) |

## Evaluation

```bash
# Swin-T
python eval_swin3d.py --model_path checkpoints/swin3d/model.pth --test_folds 1

# TimeSformer
python eval_timesformer.py --model_path checkpoints/timesformer/model.pth --test_folds 1

# Save confusion matrix image
python eval_swin3d.py --model_path model.pth --test_folds 1 --save_cm cm_fold1.png
python eval_timesformer.py --model_path model.pth --test_folds 1 --save_cm cm_fold1.png
```

Evaluation outputs per-class F1, per-class AP, macro F1, mAP, and a confusion matrix (printed as text; optionally saved as PNG with `--save_cm`).

## Method Details

**Sliding window**: A window of `window_size` frames slides over each video with the given `stride`. During training, the majority-vote label within each window is used. During evaluation, per-window softmax probabilities are averaged across all windows covering each frame (frame-wise evaluation), then optionally smoothed with `--smooth_window_size`.

**Frame sampling**: Both models receive 8 frames as input. For Swin-T, the 16-frame window is processed through the backbone's temporal pooling (output T=8). For TimeSformer, 8 frames are uniformly sampled from the 16-frame window before being fed to the model.

**Augmentation** (training only): random Gaussian blur on a subset of frames + temporal dropout (replacing frames with neighbors).
