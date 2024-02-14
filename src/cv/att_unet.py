from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from wingman import NeuralBlocks

from .attention import SelfAttention


class EnsembleAttUNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        inner_channels: list[int],
        activation: str,
        att_num_heads: int,
        num_att_module: int,
        num_ensemble: int,
    ):
        """A simple Attention UNet.

        Args:
            in_channels (int): number of channels at the input
            out_channels (int): number of channels at the output
            inner_channels (list[int]): channel descriptions for the downsampling conv net
            activation (str): activation used in all inner layers
            att_num_heads (int): number of attention heads per attention module
            num_att_module (int): number of attention modules per downscale
            num_ensemble (int): number of networks in the ensemble
        """
        super().__init__()

        self.models = nn.ModuleList(
            [
                AttUNet(
                    in_channels,
                    out_channels,
                    inner_channels,
                    activation,
                    att_num_heads,
                    num_att_module,
                )
                for _ in range(num_ensemble)
            ]
        )
        self.quantize_size = self.models[0].quantize_size

        # mean filter for uncertainty edges
        self.mean_filter = torch.nn.Conv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=3,
            padding=(3 // 2),
            bias=False,
            padding_mode="reflect",
            groups=1,
        )
        self.mean_filter.weight = torch.nn.Parameter(
            torch.ones_like(self.mean_filter.weight) / (3**2)
        )
        self.mean_filter.weight.requires_grad = False

        # this is the thing
        self.mean_filter_size = 11
        self.register_buffer(
            "mean_filter_kernel",
            torch.ones((1, 1, self.mean_filter_size, self.mean_filter_size))
            / (self.mean_filter_size**2),
            persistent=False,
        )
        self.mean_filter_kernel: torch.Tensor

    def forward(self, x):
        y = torch.stack([f(x) for f in self.models], dim=1)

        if self.training:
            return y
        else:
            # prediction is wherever it's positive, y is now [num_ensemble, B, C, H, W]
            # take the mean over the ensemble dimension, y is now [B, C, H, W] and in [0, 1]
            y = y[0] > y[1]
            y = y.sum(dim=0) / y.shape[0]

            # prediction is wherever there's a prediction
            prediction = y == 1.0

            # compute entropy of prediction, entropy is [B, C, H, W]
            # 4x(1-x) is almost equivalent for binary entropy function (-xlog_2(x) - (1-x)log_2(1-x))
            entropy = 4 * y * (1 - y)

            # take the mean of entropy over the number of labels we have to get a meaningful estimate
            entropy = entropy.mean(dim=-3, keepdim=True)

            # mean filter for perplexity
            entropy = F.conv2d(
                entropy, self.mean_filter_kernel, padding=self.mean_filter_size // 2
            )

            return prediction, entropy


class AttUNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        inner_channels: list[int],
        activation: str,
        att_num_heads: int,
        num_att_module: int,
    ):
        """A simple Attention UNet.

        Args:
            in_channels (int): number of channels at the input
            out_channels (int): number of channels at the output
            inner_channels (list[int]): channel descriptions for the downsampling conv net
            activation (str): activation used in all inner layers
            att_num_heads (int): number of attention heads per attention module
            num_att_module (int): number of attention modules per downscale
        """
        super().__init__()

        # size of image must be multiples of this number
        self.quantize_size = 2 ** (len(inner_channels) - 1)
        self.out_channels = out_channels

        # ingest and outgest layers before and after unet
        self.ingest = Plain(in_channels, inner_channels[0])
        self.outgest = Plain(inner_channels[0], out_channels * 2)

        # store the downs and ups in lists
        self.down_list = nn.ModuleList()
        self.up_list = nn.ModuleList()

        # dynamically allocate the down and up list
        for i in range(len(inner_channels) - 1):
            self.down_list.append(
                Down(inner_channels[i], inner_channels[i + 1], activation)
            )
            self.up_list.append(
                Up(inner_channels[-i - 1], inner_channels[-i - 2], activation)
            )

        # init attention modules
        self.attention = nn.Sequential(
            *[
                SelfAttention(inner_channels[-1], att_num_heads, context_length=256)
                for _ in range(num_att_module)
            ]
        )

    def forward(self, x):
        x = self.ingest(x)
        intermediates = [x := f(x) for f in self.down_list]
        x = self.attention(x)
        for f, intermediate in zip(self.up_list, reversed(intermediates)):
            x = f(x + intermediate)
        y = F.softplus(self.outgest(x)) + 1.0

        # output here is [pos_neg, B, C, H, W]
        return torch.stack(
            [y[..., : self.out_channels, :, :], y[..., self.out_channels :, :, :]],
            dim=0,
        )


def Plain(in_channel, out_channel):
    """
    Just a plain ol convolution
    """
    channels = [in_channel, out_channel]
    kernels = [3]
    pooling = [0]
    activation = ["identity"] * len(kernels)

    return NeuralBlocks.generate_conv_stack(
        channels, kernels, pooling, activation, norm="batch"
    )


def Down(in_channel, out_channel, activation):
    """
    batchnorm -> conv -> activation -> maxpool
    downscales input by 2
    """
    channels = [in_channel, out_channel]
    kernels = [3]
    pooling = [2]
    activation = [activation] * len(kernels)

    return NeuralBlocks.generate_conv_stack(
        channels, kernels, pooling, activation, norm="batch"
    )


def Up(in_channel, out_channel, activation):
    """
    batchnorm -> deconv -> activation
    upscales input by 2
    """
    channels = [in_channel, out_channel]
    kernels = [4]
    padding = [1]
    stride = [2]
    activation = [activation] * len(kernels)

    return NeuralBlocks.generate_deconv_stack(
        channels, kernels, padding, stride, activation, norm="batch"
    )
