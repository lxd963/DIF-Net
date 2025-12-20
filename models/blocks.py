import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import CoordAttMeanMax, SEBlock


class DepthwiseSeparableConv(nn.Module):
    """Depthwise Separable Convolution"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels,
            kernel_size, stride=stride, padding=padding,
            groups=in_channels, bias=bias
        )
        self.pointwise = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=1, bias=bias
        )

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class LocalBranch(nn.Module):
    """Local Branch with Multi-scale Depthwise Separable Convolutions"""
    def __init__(self, channels):
        super().__init__()
        # 3x3 depthwise separable branch
        self.dw3x3_branch = nn.Sequential(
            DepthwiseSeparableConv(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            DepthwiseSeparableConv(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels)
        )
        
        # 5x5 depthwise separable branch
        self.dw5x5_branch = nn.Sequential(
            DepthwiseSeparableConv(channels, channels, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            DepthwiseSeparableConv(channels, channels, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(channels)
        )
        
        # SE attention
        self.se = SEBlock(2 * channels)
        
        # Channel reduction after concatenation
        self.combine_conv = nn.Sequential(
            nn.Conv2d(2 * channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels)
        )
        
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        # Process two branches in parallel
        out3x3 = self.dw3x3_branch(x)
        out5x5 = self.dw5x5_branch(x)
        
        # Concatenate along channel dimension
        combined = torch.cat([out3x3, out5x5], dim=1)
        
        # Apply SE attention
        combined = self.se(combined)
        
        # Reduce channels with 1x1 convolution
        out = self.combine_conv(combined)
        
        # Residual connection
        out += identity
        return self.relu(out)


class ResidualBlockCoordAtt(nn.Module):
    """Residual Block with Coordinate Attention"""
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=False)
        self.att = CoordAttMeanMax(out_channels, out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)
        out = self.att(out)
        return out