import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import LocalBranch


class FourierUnit(nn.Module):
    """Fourier Unit for Frequency Domain Processing"""
    def __init__(self, in_channels, out_channels, groups=1):
        super().__init__()
        self.groups = groups
        self.mlp = nn.Sequential(
            nn.Linear(in_channels*2, out_channels*2),
            nn.ReLU(inplace=True),
            nn.Linear(out_channels*2, out_channels*2)
        )
        self.bn = nn.BatchNorm2d(out_channels*2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        batch, c, h, w = x.size()
        r_size = x.size()

        ffted = torch.fft.fft2(x, norm='ortho')
        real_ffted = torch.view_as_real(ffted)
        ffted = real_ffted.permute(0,1,4,2,3).contiguous().view(batch, -1, h, w)
        
        ffted = ffted.permute(0, 2, 3, 1)
        ffted = self.mlp(ffted)
        ffted = ffted.permute(0, 3, 1, 2)
        
        ffted = self.relu(self.bn(ffted))
        ffted = ffted.view(batch, -1, 2, h, w).permute(0,1,3,4,2).contiguous()
        complex_ffted = torch.view_as_complex(ffted)
        output = torch.fft.ifft2(complex_ffted, s=r_size[2:], norm='ortho')
        return output.real


class SpectralTransform(nn.Module):
    """Spectral Transform with Fourier Units"""
    def __init__(self, in_channels, out_channels, stride=1, groups=1, enable_lfu=True):
        super().__init__()
        self.enable_lfu = enable_lfu
        self.downsample = nn.AvgPool2d(2, stride=2) if stride==2 else nn.Identity()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels//2, kernel_size=1, groups=groups, bias=False),
            nn.BatchNorm2d(out_channels//2),
            nn.ReLU(inplace=True)
        )
        self.fu = FourierUnit(out_channels//2, out_channels//2, groups)
        self.lfu = FourierUnit(out_channels//2, out_channels//2, groups) if enable_lfu else None
        self.conv2 = nn.Conv2d(out_channels//2, out_channels, kernel_size=1, groups=groups, bias=False)

    def forward(self, x):
        x = self.downsample(x)
        x = self.conv1(x)
        out = self.fu(x)

        if self.enable_lfu:
            n,c,h,w = x.shape
            split_no = 2
            split_s_h = h//split_no
            xs = torch.cat(torch.split(x[:, :c//4], split_s_h, dim=-2), dim=1)
            xs = torch.cat(torch.split(xs, split_s_h, dim=-1), dim=1)
            xs = self.lfu(xs)
            xs = xs.repeat(1,1,split_no,split_no)
        else:
            xs = 0

        return self.conv2(x + out + xs)


class FFC(nn.Module):
    """Fast Fourier Convolution Block"""
    def __init__(self, in_channels, out_channels, kernel_size, ratio_gin=0.75, ratio_gout=0.75,
                 stride=1, padding=0, dilation=1, groups=1, bias=False, enable_lfu=True):
        super().__init__()
        in_cg = max(1, int(in_channels * ratio_gin))
        in_cl = in_channels - in_cg
        out_cg = max(1, int(out_channels * ratio_gout))
        out_cl = out_channels - out_cg

        self.in_cl = in_cl
        self.in_cg = in_cg

        self.local_branch1 = LocalBranch(in_cl) if (in_cl > 0 and out_cl > 0) else None
        self.local_branch2 = LocalBranch(in_cl) if (in_cl > 0 and out_cg > 0) else None
        self.convg2g = SpectralTransform(in_cg, out_cg, stride, groups, enable_lfu) if (in_cg > 0 and out_cg > 0) else None
        self.conv_l2l = nn.Conv2d(in_cl, out_cl, 1, bias=bias) if (in_cl > 0 and out_cl > 0) else None
        self.conv_l2g = nn.Conv2d(in_cl, out_cg, 1, bias=bias) if (in_cl > 0 and out_cg > 0) else None
        
        self.fuse_conv = nn.Conv2d(out_cl + out_cg, out_cl + out_cg, kernel_size=1) if out_cl + out_cg > 0 else None
        
        self.downsample = nn.AvgPool2d(2,2) if stride==2 else nn.Identity()

    def forward(self, x):
        if not isinstance(x, tuple):
            if self.in_cl > 0 and self.in_cg > 0:
                x_l, x_g = torch.split(x, [self.in_cl, self.in_cg], dim=1)
            elif self.in_cl > 0:
                x_l, x_g = x, 0
            elif self.in_cg > 0:
                x_l, x_g = 0, x
            else:
                x_l, x_g = 0, 0
        else:
            x_l, x_g = x
        
        if isinstance(x_l, torch.Tensor):
            x_l = self.downsample(x_l)

        out_l = self.local_branch1(x_l) if (self.local_branch1 and isinstance(x_l, torch.Tensor) and x_l.shape[1] == self.in_cl) else 0
        out_l2g = self.local_branch2(x_l) if (self.local_branch2 and isinstance(x_l, torch.Tensor) and x_l.shape[1] == self.in_cl) else 0
        
        if self.conv_l2l and isinstance(out_l, torch.Tensor) and out_l.shape[1] > 0:
            out_l = self.conv_l2l(out_l)
        if self.conv_l2g and isinstance(out_l2g, torch.Tensor) and out_l2g.shape[1] > 0:
            out_l2g = self.conv_l2g(out_l2g)
        
        if self.convg2g and isinstance(x_g, torch.Tensor):
            global_out = self.convg2g(x_g)
            out_g = out_l2g + global_out if isinstance(out_l2g, torch.Tensor) else global_out
        else:
            out_g = out_l2g

        if self.fuse_conv is not None:
            parts = []
            out_cl_channels = 0
            if isinstance(out_l, torch.Tensor) and out_l.numel() > 0:
                parts.append(out_l)
                out_cl_channels = out_l.shape[1]
            
            out_cg_channels = 0
            if isinstance(out_g, torch.Tensor) and out_g.numel() > 0:
                parts.append(out_g)
                out_cg_channels = out_g.shape[1]

            if parts:
                combined = torch.cat(parts, dim=1)
                fused = self.fuse_conv(combined)
                
                if out_cl_channels > 0 and out_cg_channels > 0:
                    out_l, out_g = torch.split(fused, [out_cl_channels, out_cg_channels], dim=1)
                elif out_cl_channels > 0:
                    out_l, out_g = fused, 0
                elif out_cg_channels > 0:
                    out_l, out_g = 0, fused
            else:
                out_l, out_g = 0, 0

        return out_l, out_g


class FFC_BN_ACT(nn.Module):
    """FFC with BatchNorm and Activation"""
    def __init__(self, in_channels, out_channels, kernel_size,
                 ratio_gin=0.75, ratio_gout=0.75, stride=1, padding=0,
                 groups=1, bias=False,
                 norm_layer=nn.BatchNorm2d, activation_layer=nn.ReLU,
                 enable_lfu=True):
        super().__init__()
        self.ffc = FFC(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
            ratio_gin=ratio_gin, ratio_gout=ratio_gout, stride=stride, padding=padding,
            groups=groups, bias=bias, enable_lfu=enable_lfu
        )
        out_cl = int(out_channels * (1 - ratio_gout))
        out_cg = out_channels - out_cl
        self.bn_l = nn.Identity() if ratio_gout == 1 or out_cl == 0 else norm_layer(out_cl)
        self.bn_g = nn.Identity() if ratio_gout == 0 or out_cg == 0 else norm_layer(out_cg)
        self.act_l = nn.Identity() if ratio_gout == 1 or out_cl == 0 else activation_layer(inplace=True)
        self.act_g = nn.Identity() if ratio_gout == 0 or out_cg == 0 else activation_layer(inplace=True)

    def forward(self, x):
        x_l, x_g = self.ffc(x)
        
        if isinstance(x_l, torch.Tensor) and x_l.numel() > 0:
            x_l_out = self.act_l(self.bn_l(x_l))
        else:
            x_l_out = 0
        
        if isinstance(x_g, torch.Tensor) and x_g.numel() > 0:
            x_g_out = self.act_g(self.bn_g(x_g))
        else:
            x_g_out = 0
            
        return x_l_out, x_g_out