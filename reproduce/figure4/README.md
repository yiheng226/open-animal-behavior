# Mouse Behavior Classification — Video-Level Data Efficiency Analysis

Extension of the base Video Swin-T / TimeSformer pipeline with **video-level subsampling** for data efficiency experiments.

Due to GPU nondeterminism, mixed precision training, and multi-worker data loading, retraining the model may produce slightly different results across runs even with the same random seed.

To ensure exact reproducibility of the reported metrics, we can provide the pretrained model checkpoints used in the paper upon request.

## Requirements

```bash
pip install torch torchvision transformers decord numpy pandas scikit-learn matplotlib seaborn tqdm
```

## Arguments

| Argument | Default | Description |
|---|---|---|
| `--train_data_ratio` | `1.0` | Fraction of training videos to use (0.0–1.0) |
| `--video_split_seed` | `42` | RNG seed for video selection (42，123，999) |

All other arguments are identical to the base scripts — see the base README.

## 3-Fold Cross-Validation

| Fold | `--test_folds` | `--train_folds` |
|------|----------------|-----------------|
| Fold 1 | `1` | `3 2` |
| Fold 2 | `2` | `1 3` |
| Fold 3 | `3` | `1 2` |

## Training

```bash
# Video Swin-T — 50% of training videos, 3-fold CV
python train_swin3d_ratio.py --test_folds 1 --train_folds 3 2 --train_data_ratio 0.5 --video_split_seed 42
python train_swin3d_ratio.py --test_folds 2 --train_folds 1 3 --train_data_ratio 0.5 --video_split_seed 42
python train_swin3d_ratio.py --test_folds 3 --train_folds 1 2 --train_data_ratio 0.5 --video_split_seed 42

# Custom paths
python train_swin3d_ratio.py --base_video_dir /path/to/videos --label_dir /path/to/labels

# TimeSformer — 50% of training videos, 3-fold CV
python train_timesformer_ratio.py --test_folds 1 --train_folds 3 2 --train_data_ratio 0.5 --video_split_seed 42
python train_timesformer_ratio.py --test_folds 2 --train_folds 1 3 --train_data_ratio 0.5 --video_split_seed 42
python train_timesformer_ratio.py --test_folds 3 --train_folds 1 2 --train_data_ratio 0.5 --video_split_seed 42

# Custom paths
python train_timesformer_ratio.py --base_video_dir /path/to/videos --label_dir /path/to/labels
```

## Evaluation

```bash
# Video Swin-T
python eval_swin3d_ratio.py --model_path checkpoints/swin3d_ratio/model.pth --test_folds 1

# TimeSformer
python eval_timesformer_ratio.py --model_path checkpoints/timesformer_ratio/model.pth --test_folds 1

# Save confusion matrix image
python eval_swin3d_ratio.py --model_path model.pth --test_folds 1 --save_cm cm_fold1.png
```

## Output Naming Convention

Checkpoint filenames encode the ratio and seed:

```
{model}_train_{folds}_val_{folds}_ratio{pct}_vseed{seed}_ep{N}_f1_{val}_map_{val}.pth
```

Example: `swin3d_train_3_2_val_1_ratio50_vseed42_ep5_f1_0.6123_map_0.7456.pth`

At `ratio=100` (full data), the `_vseed` suffix is omitted.
