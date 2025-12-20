import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import ResidualBlockCoordAtt
from .ffc_modules import FFC_BN_ACT
from .attention import CoordAttMeanMax


class BasicBlock(nn.Module):
    """Basic Block for FFC-based ResNet"""
    expansion = 1
    
    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 ratio_gin=0.75, ratio_gout=0.75, use_se=True, lfu=True, norm_layer=None, use_att=False):
        super().__init__()
        norm_layer = norm_layer or nn.BatchNorm2d
        width = planes
        
        self.use_att = use_att
        self.conv1 = FFC_BN_ACT(inplanes, width, 3, ratio_gin, ratio_gout,
                                stride, padding=1, norm_layer=norm_layer,
                                activation_layer=nn.ReLU, enable_lfu=lfu)
        self.conv2 = FFC_BN_ACT(width, planes*self.expansion, 3,
                                ratio_gout, ratio_gout, padding=1,
                                norm_layer=norm_layer, enable_lfu=lfu)
        
        if use_att and ratio_gout < 1.0:
            local_channels = int(planes * self.expansion * (1 - ratio_gout))
            if local_channels > 0:
                self.att = CoordAttMeanMax(local_channels, local_channels)
            else:
                self.att = nn.Identity()
        else:
            self.att = nn.Identity()
            
        self.se_block = nn.Identity()
        self.relu_l = nn.Identity() if ratio_gout==1 else nn.ReLU(inplace=True)
        self.relu_g = nn.Identity() if ratio_gout==0 else nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def _match(self, src, tgt):
        if not isinstance(tgt, torch.Tensor): return 0
        if not isinstance(src, torch.Tensor): return torch.zeros_like(tgt)
        if src.shape[2:] != tgt.shape[2:]:
            src = F.adaptive_avg_pool2d(src, tgt.shape[2:])
        if src.shape[1] != tgt.shape[1]:
            src = nn.Conv2d(src.shape[1], tgt.shape[1], 1).to(src.device)(src)
        return src
    
    def forward(self, x):
        x = x if isinstance(x, tuple) else (x,0)
        identity = x
        if self.downsample:
            identity = self.downsample(x)
            
        out = self.conv1(x)
        out = self.conv2(out)
        
        if isinstance(out, tuple):
            out_l, out_g = out
            if isinstance(out_l, torch.Tensor) and out_l.numel() > 0:
                out_l = self.att(out_l)
            out = (out_l, out_g)
        
        out_l, out_g = self.se_block(out)

        id_l, id_g = identity
        id_l = self._match(id_l, out_l)
        id_g = self._match(id_g, out_g)

        res_l = self.relu_l(out_l + id_l) if isinstance(out_l, torch.Tensor) else 0
        res_g = self.relu_g(out_g + id_g) if isinstance(out_g, torch.Tensor) else 0
            
        return res_l, res_g


class DualPathFFCResNet(nn.Module):
    """Dual-Path FFC ResNet for Heart Sound Classification"""
    def __init__(self, block, layers, num_classes=2, ratio=0.75, use_se=True,
                 lfu=True, in_channels=3, use_att=False):
        super().__init__()
        self.use_se = use_se
        self.lfu = lfu
        self.use_att = use_att
        
        # STFT pathway
        self.stft_conv1 = nn.Conv2d(in_channels, 32, 3, stride=2, padding=1, bias=False)
        self.stft_bn1 = nn.BatchNorm2d(32)
        self.stft_relu = nn.ReLU(inplace=True)
        
        # PCG pathway
        self.pcg_conv1 = nn.Conv2d(in_channels, 32, 3, stride=2, padding=1, bias=False)
        self.pcg_bn1 = nn.BatchNorm2d(32)
        self.pcg_relu = nn.ReLU(inplace=True)
        
        # STFT layers (traditional CNN)
        self.inplanes = 32
        self.stft_layer1 = self._make_stft_layer(32, layers[0], stride=1)
        self.stft_layer2 = self._make_stft_layer(64, layers[1], stride=2)
        self.stft_layer3 = self._make_stft_layer(128, layers[2], stride=2)
        self.stft_layer4 = self._make_stft_layer(256, layers[3], stride=2)
        
        # PCG layers (FFC-based)
        self.inplanes = 32
        self.pcg_layer1 = self._make_pcg_layer(32, layers[0], stride=1, ratio_gin=0, ratio_gout=ratio)
        self.pcg_layer2 = self._make_pcg_layer(64, layers[1], stride=2, ratio_gin=ratio, ratio_gout=ratio)
        self.pcg_layer3 = self._make_pcg_layer(128, layers[2], stride=2, ratio_gin=ratio, ratio_gout=ratio)
        self.pcg_layer4 = self._make_pcg_layer(256, layers[3], stride=2, ratio_gin=ratio, ratio_gout=ratio)
        
        # Feature fusion
        self.final_adjust_layer = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        # Classification head
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)
        
        # Initialize weights
        self._initialize_weights()

    def _make_pcg_layer(self, planes, blocks, stride=1, ratio_gin=0.75, ratio_gout=0.75):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = FFC_BN_ACT(
                self.inplanes, planes, 1,
                ratio_gin, ratio_gout, stride=stride, enable_lfu=True
            )
        layers = [
            BasicBlock(self.inplanes, planes, stride, downsample, 
                       ratio_gin=ratio_gin, ratio_gout=ratio_gout, 
                       use_se=self.use_se, lfu=self.lfu, use_att=self.use_att)
        ]
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(BasicBlock(
                self.inplanes, planes, 
                ratio_gin=ratio_gout, ratio_gout=ratio_gout, 
                use_se=self.use_se, lfu=self.lfu, use_att=self.use_att
            ))
        return nn.Sequential(*layers)
    
    def _make_stft_layer(self, planes, blocks, stride=1):
        layers = []
        layers.append(ResidualBlockCoordAtt(self.inplanes, planes, stride=stride))
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(ResidualBlockCoordAtt(self.inplanes, planes))
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, stft_input, pcg_input):
        # STFT pathway
        stft_x = self.stft_relu(self.stft_bn1(self.stft_conv1(stft_input)))
        stft_x = self.stft_layer1(stft_x)
        stft_x = self.stft_layer2(stft_x)
        stft_x = self.stft_layer3(stft_x)
        stft_x = self.stft_layer4(stft_x)
        
        # PCG pathway
        pcg_x = self.pcg_relu(self.pcg_bn1(self.pcg_conv1(pcg_input)))
        pcg_x = self.pcg_layer1(pcg_x)
        pcg_x = self.pcg_layer2(pcg_x)
        pcg_x = self.pcg_layer3(pcg_x)
        pcg_x = self.pcg_layer4(pcg_x)
        
        # Combine PCG features (handle tuple output from FFC)
        if isinstance(pcg_x, tuple):
            pcg_parts = [p for p in pcg_x if isinstance(p, torch.Tensor) and p.numel() > 0]
            if pcg_parts:
                pcg_combined = torch.cat(pcg_parts, dim=1)
            else:
                pcg_combined = torch.zeros_like(stft_x)
        else:
            pcg_combined = pcg_x if isinstance(pcg_x, torch.Tensor) else torch.zeros_like(stft_x)

        # Fuse dual-path features
        fused_x = torch.cat([stft_x, pcg_combined], dim=1)
        fused_x = self.final_adjust_layer(fused_x)
        
        # Classification
        x = self.avgpool(fused_x)
        features = torch.flatten(x, 1)
        return self.fc(features), features


def dualpath_ffc_resnet18(**kwargs):
    """Create a DualPathFFCResNet-18 model"""
    return DualPathFFCResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
