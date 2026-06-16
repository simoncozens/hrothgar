import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import numpy as np
import torch


def dispatcher(dispatch_fn):
    def decorator(key, *args):
        if callable(key):
            return key
        if key is None:
            key = "none"
        return dispatch_fn(key, *args)

    return decorator


@dispatcher
def norm_dispatch(norm):
    return {
        "none": nn.Identity,
        "in": partial(nn.InstanceNorm2d, affine=False),
        "bn": nn.BatchNorm2d,
        "group": partial(nn.GroupNorm, num_groups=32, eps=1e-6, affine=True),
        "batch": nn.SyncBatchNorm,
    }[norm.lower()]


@dispatcher
def w_norm_dispatch(w_norm):
    return {"spectral": spectral_norm, "none": lambda x: x}[w_norm.lower()]


@dispatcher
def activ_dispatch(activ):
    return {
        "none": nn.Identity,
        "relu": nn.ReLU,
        "lrelu": partial(nn.LeakyReLU, negative_slope=0.2),
    }[activ.lower()]


@dispatcher
def pad_dispatch(pad_type):
    return {
        "zero": nn.ZeroPad2d,
        "replicate": nn.ReplicationPad2d,
        "reflect": nn.ReflectionPad2d,
    }[pad_type.lower()]


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        norm="none",
        w_norm="none",
        activ="relu",
        pad_type="zero",
        bias=True,
        upsample=False,
        downsample=False,
        dropout=0.0,
    ):
        super().__init__()
        assert not (
            upsample and downsample
        ), "Cannot upsample and downsample at the same time"
        if kernel_size == 1:
            assert padding == 0, "padding must be 0 for kernel_size=1"

        self.in_channels = in_channels
        self.out_channels = out_channels

        norm_type = norm
        if norm_type == "group":
            norm = norm_dispatch(norm_type)(num_channels=in_channels)
        elif norm_type in ["bn", "batch", "in"]:
            norm = norm_dispatch(norm_type)(num_features=in_channels)
        else:
            norm = norm_dispatch(norm_type)()

        w_norm = w_norm_dispatch(w_norm)
        activ = activ_dispatch(activ)
        pad_type = pad_dispatch(pad_type)

        self.upsample = upsample
        self.downsample = downsample

        self.norm = norm
        self.activ = activ()

        if dropout > 0.0:
            self.dropout = nn.Dropout2d(dropout)

        self.pad = pad_type(padding)
        self.conv = w_norm(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, bias=bias)
        )

    def forward(self, x):

        x = self.norm(x)  # Normalization [N, In_C, H, W]
        x = self.activ(x)  # Activation [N, In_C, H, W]
        if self.upsample:
            x = F.interpolate(x, scale_factor=2)  # [N, In_C, H*2, W*2]

        if hasattr(self, "dropout"):
            x = self.dropout(x)  # Dropout [N, In_C, H, W] or [N, In_C, H*2, W*2]

        x = self.conv(self.pad(x))  # Convolution [N, Out_C, H', W']

        if self.downsample:
            x = F.avg_pool2d(x, kernel_size=2)  # [N, Out_C, H'/2, W'/2]

        return x  # [N, Out_C, H', W'] or [N, Out_C, H'/2, W'/2]


class ResBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        padding=1,
        norm="none",
        w_norm="none",
        activ="relu",
        pad_type="zero",
        bias=True,
        upsample=False,
        downsample=False,
        dropout=0.0,
        scale_var=False,
    ):
        super().__init__()
        assert not (
            upsample and downsample
        ), "Cannot upsample and downsample at the same time"

        w_norm = w_norm_dispatch(w_norm)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.upsample = upsample
        self.downsample = downsample
        self.scale_var = scale_var

        self.conv1 = ConvBlock(
            in_channels,
            out_channels,
            kernel_size,
            1,
            padding,
            norm,
            w_norm,
            activ,
            pad_type,
            upsample=upsample,
            dropout=dropout,
        )
        self.conv2 = ConvBlock(
            out_channels,
            out_channels,
            kernel_size,
            1,
            padding,
            norm,
            w_norm,
            activ,
            pad_type,
            dropout=dropout,
        )

        # XXX upsample / downsample needs skip conv?
        if in_channels != out_channels or upsample or downsample:
            self.skip = w_norm(nn.Conv2d(in_channels, out_channels, 1))

    def forward(self, x):
        out = x
        out = self.conv1(out)  # [N, Out_C, H, W] or [N, Out_C, 2*H, 2*W]
        out = self.conv2(out)  # [N, Out_C, H, W]

        if self.downsample:
            out = F.avg_pool2d(out, 2)  # [N, Out_C, H/2, W/2]

        # skip-convolution
        if hasattr(self, "skip"):
            if self.upsample:
                x = F.interpolate(x, scale_factor=2)  # [N, In_C, 2*H, 2*W]
            x = self.skip(x)  # [N, Out_C, H, W] or [N, Out_C, 2*H, 2*W]

            if self.downsample:
                x = F.avg_pool2d(x, 2)  # [N, Out_C, H/2, W/2]

        out = out + x
        if self.scale_var:
            out = out / np.sqrt(2)

        return out


class AttentionBlock(nn.Module):
    """
    A self-attention block for 2D feature maps.
    """

    def __init__(self, in_channels, norm_type="group"):
        super().__init__()
        if norm_type == "group":
            norm = norm_dispatch(norm_type)(num_channels=in_channels)
        elif norm_type in ["bn", "batch", "in"]:
            norm = norm_dispatch(norm_type)(num_features=in_channels)
        else:
            norm = norm_dispatch(norm_type)()

        self.norm = norm
        self.q = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.k = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.v = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.proj_out = nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )

    def forward(self, x):
        h_ = x  # [b, c, h ,w]
        h_ = self.norm(h_)  # [b, c, h ,w]
        q = self.q(h_)  # [b,c,h,w]
        k = self.k(h_)  # [b,c,h,w]
        v = self.v(h_)  # [b,c,h,w]

        # compute attention
        b, c, h, w = q.shape
        q = q.reshape(b, c, h * w)  # [b, c ,h*w]
        q = q.permute(0, 2, 1)  # [b, h*w, c]
        k = k.reshape(b, c, h * w)  # [b, c, h*w]
        w_ = torch.bmm(
            q, k
        )  # [b,h*w,h*w]    w[b,i,j]=sum_c q[b,i,c]k[b,c,j] => i's attention on j
        w_ = w_ * (int(c) ** (-0.5))
        w_ = F.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b, c, h * w)  # [b, c, h*w]
        w_ = w_.permute(0, 2, 1)  # b,hw,hw (first hw of k, second of q)
        h_ = torch.bmm(v, w_)  # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = h_.reshape(b, c, h, w)

        h_ = self.proj_out(h_)

        return x + h_


def spectral_norm(module):
    """init & apply spectral norm"""
    nn.init.xavier_uniform_(module.weight, 2**0.5)
    if hasattr(module, "bias") and module.bias is not None:
        module.bias.data.zero_()

    return nn.utils.spectral_norm(module)


def nonlinearity(x):
    # swish
    return x * torch.sigmoid(x)
