# Modified from:
#   taming-transformers: https://github.com/CompVis/taming-transformers
#   maskgit: https://github.com/google-research/maskgit
from dataclasses import dataclass, field
from typing import List
from typing import Union, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange, pack, unpack




class CausalConv3d(nn.Module):
    def __init__(
        self,
        chan_in: int = 1,
        chan_out: int = 1,
        kernel_size: Union[int, Tuple[int, int, int]] = 3,
        pad_mode: str = "constant",
        **kwargs,
    ):
        super().__init__()
        kernel_size = cast_tuple(kernel_size, 3)

        time_kernel_size, height_kernel_size, width_kernel_size = kernel_size

        assert is_odd(height_kernel_size) and is_odd(width_kernel_size)

        dilation = kwargs.pop("dilation", 1)
        stride = kwargs.pop("stride", 1)
        time_stride = kwargs.pop("time_stride", 1)
        time_dilation = kwargs.pop("time_dilation", 1)
        padding = kwargs.pop("padding", 1)

        self.pad_mode = pad_mode
        time_pad = time_dilation * (time_kernel_size - 1) + (1 - time_stride)
        self.time_pad = time_pad

        self.spatial_pad = (padding, padding, padding, padding)

        stride = (time_stride, stride, stride)
        dilation = (time_dilation, dilation, dilation)
        self.conv3d = nn.Conv3d(
            chan_in,
            chan_out,
            kernel_size,
            stride=stride,
            dilation=dilation,
            **kwargs,
        )

    def _replication_pad(self, x: torch.Tensor) -> torch.Tensor:
        x_prev = x[:, :, :1, ...].repeat(1, 1, self.time_pad, 1, 1)
        x = torch.cat([x_prev, x], dim=2)
        padding = self.spatial_pad + (0, 0)
        return F.pad(x, padding, mode=self.pad_mode, value=0.0)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 获取输入的最后两维度大小
        input_size = x.shape[-2:]  
        # 应用 Padding
        x = self._replication_pad(x)
        
        
        # 通过 Conv3d 层
        output = self.conv3d(x)
        
        # 获取输出的最后两维度大小
        output_size = output.shape[-2:]
        
        # 检查输入和输出最后两维度是否相等
        if input_size != output_size:
            raise ValueError(
                f"Mismatch between input and output sizes in the last two dimensions: "
                f"input size {input_size}, output size {output_size}."
                f"padding {self.spatial_pad} "
                f"te"
            )
        
        return output
