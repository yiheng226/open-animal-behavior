import torch
import torch.nn as nn


class MLPHead_CLS(nn.Module):
    def __init__(self, in_features, num_classes, hidden_dim, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(in_features)
        self.fc1 = nn.Linear(in_features, hidden_dim)
        self.relu = nn.ReLU(True)
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        return self.fc2(self.drop(self.relu(self.fc1(self.norm(x)))))


class MLPHead_TemporalMean(nn.Module):
    def __init__(self, in_features, num_classes, hidden_dim, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(in_features)
        self.fc1 = nn.Linear(in_features, hidden_dim)
        self.relu = nn.ReLU(True)
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = torch.mean(x, dim=1)
        return self.fc2(self.drop(self.relu(self.fc1(self.norm(x)))))


class CustomTimeSformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        from transformers import TimesformerModel
        self.backbone = TimesformerModel.from_pretrained(cfg["backbone"]["pretrained"])
        self.head = MLPHead_CLS(
            cfg["head"]["in_features"],
            cfg["num_classes"],
            cfg["head"]["hidden_dim"],
            cfg["head"]["dropout"],
        )

    def forward(self, x):
        x = x.permute(0, 2, 1, 3, 4)
        out = self.backbone(pixel_values=x)
        return self.head(out.last_hidden_state[:, 0])


class CustomSwin3D(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        from torchvision.models.video import swin3d_t, Swin3D_T_Weights
        self.model = swin3d_t(weights=Swin3D_T_Weights.DEFAULT)
        self.model.head = nn.Identity()
        self.model.avgpool = nn.Identity()
        self.head = MLPHead_TemporalMean(
            cfg["head"]["in_features"],
            cfg["num_classes"],
            cfg["head"]["hidden_dim"],
            cfg["head"]["dropout"],
        )

    def forward(self, x):
        x = self.model.patch_embed(x)
        x = self.model.pos_drop(x)
        x = self.model.features(x)
        x = self.model.norm(x)
        x = x.mean(dim=(2, 3))
        return self.head(x)


def build_model_from_config(cfg):
    """Build model based on backbone name in config."""
    name = cfg["backbone"]["name"]
    if name == "TimesformerModel" or "timesformer" in name.lower():
        return CustomTimeSformer(cfg)
    elif name == "CustomSwin3D" or "swin" in name.lower():
        return CustomSwin3D(cfg)
    else:
        raise ValueError(f"Unknown backbone: {name}")
