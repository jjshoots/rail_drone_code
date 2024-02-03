from __future__ import annotations

import torch
import torch.nn as nn


class MeanFilter(nn.Module):
    def __init__(self, kernel_size: int = 3):
        super().__init__()
        self.mean_filter = torch.nn.Conv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=kernel_size,
            padding=(kernel_size // 2),
            bias=False,
            padding_mode="reflect",
            groups=1,
        )
        self.mean_filter.weight = torch.nn.Parameter(
            torch.ones_like(self.mean_filter.weight) / (kernel_size**2)
        )
        self.mean_filter.weight.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = []
        for i in range(x.shape[1]):
            output.append(self.mean_filter(x[:, [i], ...]))

        return torch.cat(output, dim=1)
