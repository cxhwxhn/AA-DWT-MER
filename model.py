from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models


def conv3x3(in_channels, out_channels):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=True,
    )


def conv1x1(in_channels, out_channels):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        padding=0,
        bias=False,
    )


class ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1):
        super().__init__()
        if kernel_size == 1:
            conv = conv1x1(in_channels, out_channels)
        elif kernel_size == 3:
            conv = conv3x3(in_channels, out_channels)
        else:
            raise ValueError(f"Unsupported kernel size: {kernel_size}")

        self.block = nn.Sequential(
            conv,
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class ChannelShuffle(nn.Module):
    def __init__(self, groups):
        super().__init__()
        self.groups = groups

    def forward(self, x):
        batch, channels, height, width = x.size()
        if channels % self.groups != 0:
            raise ValueError(
                f"channels={channels} must be divisible by groups={self.groups}"
            )

        x = x.view(batch, self.groups, channels // self.groups, height, width)
        x = x.transpose(1, 2).contiguous()
        return x.view(batch, channels, height, width)


class SymmetricChannelFlow(nn.Module):
    

    def __init__(self, channels, groups=4):
        super().__init__()
        if channels % groups != 0:
            raise ValueError(f"channels={channels} must be divisible by groups={groups}")

        self.channels = channels
        self.groups = groups
        self.block_channels = channels // groups
        self.shuffle = ChannelShuffle(groups)

        self.left_convs = nn.ModuleList(
            [conv3x3(self.block_channels, self.block_channels) for _ in range(groups)]
        )
        self.right_convs = nn.ModuleList(
            [conv3x3(self.block_channels, self.block_channels) for _ in range(groups)]
        )
        self.fuse = ConvBNReLU(channels * 2, channels, kernel_size=1)

    def _left_to_right_flow(self, blocks):
        features = []
        running = None
        for block, conv in zip(blocks, self.left_convs):
            running = block if running is None else running + block
            features.append(conv(running))
        return features

    def _right_to_left_flow(self, blocks):
        features = []
        running = None
        for block, conv in zip(reversed(blocks), reversed(self.right_convs)):
            running = block if running is None else running + block
            features.append(conv(running))
        features.reverse()
        return features

    def forward(self, x):
        shuffled = self.shuffle(x)
        blocks = torch.chunk(shuffled, self.groups, dim=1)

        left_features = self._left_to_right_flow(blocks)
        right_features = self._right_to_left_flow(blocks)
        channel_features = torch.cat(left_features + right_features, dim=1)
        return self.fuse(channel_features)


class SpatialFlow(nn.Module):
  

    def __init__(self):
        super().__init__()
        self.attention = nn.Sequential(
            conv3x3(2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        avg_map = torch.mean(x, dim=1, keepdim=True)
        max_map, _ = torch.max(x, dim=1, keepdim=True)
        spatial_descriptor = torch.cat([avg_map, max_map], dim=1)
        spatial_weight = self.attention(spatial_descriptor)
        return x * spatial_weight


class ChannelSpatialEnhancement(nn.Module):
 

    def __init__(self, channels, groups=4):
        super().__init__()
        self.channel_flow = SymmetricChannelFlow(channels, groups=groups)
        self.spatial_flow = SpatialFlow()
        self.output_fusion = ConvBNReLU(channels * 2, channels, kernel_size=1)

    def forward(self, x):
        channel_feature = self.channel_flow(x)
        spatial_feature = self.spatial_flow(x)
        enhanced_feature = torch.cat([channel_feature, spatial_feature], dim=1)
        return x + self.output_fusion(enhanced_feature)


class ResNet18Backbone(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.backbone = self._build_resnet18(pretrained)
        self.features = nn.Sequential(*list(self.backbone.children())[:-2])

    @staticmethod
    def _build_resnet18(pretrained):
        local_weights = Path(__file__).resolve().parent / "pretrained" / "resnet18-f37072fd.pth"
        if pretrained and local_weights.exists():
            try:
                model = models.resnet18(weights=None)
            except TypeError:
                model = models.resnet18(pretrained=False)
            state_dict = torch.load(str(local_weights), map_location="cpu")
            model.load_state_dict(state_dict)
            print("Loaded local ResNet18 weights: {}".format(local_weights))
            return model

        try:
            weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            return models.resnet18(weights=weights)
        except (AttributeError, TypeError):
            return models.resnet18(pretrained=pretrained)

    def forward(self, x):
        return self.features(x)


class ClassificationHead(nn.Module):
    def __init__(self, in_channels, num_classes, dropout=0.3):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(in_channels, num_classes),
        )

    def forward(self, x):
        x = self.pool(x)
        return self.classifier(x)


class AADWTMER(nn.Module):
    def __init__(self, num_classes, pretrained=True, groups=4, dropout=0.3):
        super().__init__()
        self.backbone = ResNet18Backbone(pretrained=pretrained)
        self.enhance = ChannelSpatialEnhancement(512, groups=groups)
        self.head = ClassificationHead(
            in_channels=512,
            num_classes=num_classes,
            dropout=dropout,
        )

    def forward_features(self, x):
        x = self.backbone(x)
        x = self.enhance(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        return self.head(x)
