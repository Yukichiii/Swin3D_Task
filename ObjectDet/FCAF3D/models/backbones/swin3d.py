"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT License.
"""
import math
import numpy as np
import torch
import torch.nn as nn
from mmdet3d.models.builder import BACKBONES

from torch_scatter import scatter_softmax, scatter_sum
from timm.models.layers import DropPath, trunc_normal_

import MinkowskiEngine as ME
from MinkowskiEngine import SparseTensor
from numpy import dtype, product

from Swin3D.modules.mink_layers import (SparseTensorLayerNorm, SparseTensorLinear, 
    MinkConvBNRelu, MinkDeConvBNRelu, MinkResBlock, assign_feats)
from Swin3D.modules.swin3d_layers import GridDownsample, GridKNNDownsample, BasicLayer

@BACKBONES.register_module()
class Swin3DEncoder_RGB(nn.Module):
    def __init__(self, in_channels=3, depths=[2,2,2,2,2], channels = [16, 48, 96, 192, 384], num_heads=[1, 3, 6, 12, 24], window_size=5, knn_down=True, stem_transformer=False, num_out=4):
        super().__init__()
        self.num_out = num_out
        num_layers = len(depths)
        if isinstance(window_size, list):
            window_sizes = window_size
        else:
            window_sizes = [window_size] * num_layers
        quant_sizes = [4] * num_layers
        rel_query, rel_key, rel_value = True, True, True
        drop_path_rate = 0.3
        if knn_down:
            downsample = GridKNNDownsample
        else:
            downsample = GridDownsample
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        if stem_transformer:
            self.stem_layer = MinkConvBNRelu(
                    in_channels=in_channels,
                    out_channels=channels[0],
                    kernel_size=3,
                    stride=1,
                )
            self.layer_start = 0
        else:
            self.stem_layer = nn.Sequential(              
                MinkConvBNRelu(
                    in_channels=in_channels,
                    out_channels=channels[0],
                    kernel_size=3,
                    stride=1,
                ),
                MinkResBlock(
                    in_channels=channels[0],
                    out_channels=channels[0]
                )
            )
            self.downsample = downsample(
                        channels[0],
                        channels[1],
                        kernel_size=2,
                        stride=2
            )
            self.layer_start = 1
        self.layers = nn.ModuleList([
            BasicLayer(
                dim=channels[i], 
                depth=depths[i], 
                num_heads=num_heads[i], 
                window_size=window_sizes[i],
                quant_size=quant_sizes[i],
                drop_path=dpr[sum(depths[:i]):sum(depths[:i+1])], 
                downsample=downsample if i < num_layers-1 else None,
                down_stride=2,
                out_channels=channels[i+1] if i < num_layers-1 else None,
                cRSE='XYZ_RGB',
                fp16_mode=1,
                ) for i in range(self.layer_start, num_layers)])

        self.init_weights()

    def forward(self, sp):
        self.device = sp.F.device
        voxel_offset = sp.F[:, 3:]
        coords = sp.C.clone().float()
        coords[:, 1:] += voxel_offset
        colors = sp.F[:, :3].float()
        colors = (colors * 2 - 1) / 1.001
        coords_sp = SparseTensor(features=torch.cat([coords, colors], dim=1), coordinate_map_key=sp.coordinate_map_key, coordinate_manager=sp.coordinate_manager)
        sp = SparseTensor(sp.F.float(), coordinate_map_key=sp.coordinate_map_key, coordinate_manager=sp.coordinate_manager)
        sp = self.stem_layer(sp)
        if self.layer_start > 0:
            sp, coords_sp = self.downsample(sp, coords_sp)

        sp_stack = []
        for i, layer in enumerate(self.layers):
            sp, sp_down, coords_sp = layer(sp, coords_sp)
            sp_stack.append(sp)
            assert (coords_sp.C == sp_down.C).all()
            sp = sp_down

        # for sp in sp_stack:
        #     print(sp.coordinate_map_key.get_tensor_stride(), sp.F.shape)
        return sp_stack[-self.num_out:]

    def init_weights(self):
        """Initialize the weights in backbone.
        """

        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

            elif isinstance(m, ME.MinkowskiConvolution):
                ME.utils.kaiming_normal_(
                    m.kernel, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, ME.MinkowskiBatchNorm):
                nn.init.constant_(m.bn.weight, 1)
                nn.init.constant_(m.bn.bias, 0)

        self.apply(_init_weights)
