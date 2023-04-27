import os
import torch
import torch.nn as nn
from Swin3D.models import Swin3DUNet
from MinkowskiEngine import SparseTensor


class Swin3D(nn.Module):
    def __init__(
        self,
        depths,
        channels,
        num_heads,
        window_sizes,
        up_k,
        quant_sizes,
        drop_path_rate=0.2,
        num_layers=4,
        num_classes=13,
        stem_transformer=False,
        upsample="deconv",
        down_stride=2,
        knn_down=True,
        signal=True,
        in_channels=6,
        use_offset=False,
        fp16_mode=2,
    ):
        super().__init__()
        self.signal = signal
        self.use_offset = use_offset
        self.backbone = Swin3DUNet(
            depths,
            channels,
            num_heads,
            window_sizes,
            quant_sizes,
            up_k=up_k,
            drop_path_rate=drop_path_rate,
            num_classes=num_classes,
            num_layers=num_layers,
            stem_transformer=stem_transformer,
            upsample=upsample,
            first_down_stride=down_stride,
            knn_down=knn_down,
            in_channels=in_channels,
            cRSE="XYZ_RGB_NORM",
            fp16_mode=fp16_mode,
        )

    def forward(self, feats, xyz, batch):
        self.device = feats.device
        coords = torch.cat([batch.unsqueeze(-1), xyz], dim=-1)
        if self.signal:
            if feats.shape[1] > 3:
                if self.use_offset:
                    feats[:, -3:] = xyz - xyz.int()
            sp = SparseTensor(feats.float(), coords.int(), device=self.device)
        else:
            sp = SparseTensor(
                torch.ones_like(feats).float(), coords.int(), device=self.device
            )
        colors = feats[:, 0:3] / 1.001
        normals = feats[:, 3:6] / 1.001
        coords_sp = SparseTensor(
            features=torch.cat([coords, colors, normals], dim=1),
            coordinate_map_key=sp.coordinate_map_key,
            coordinate_manager=sp.coordinate_manager,
        )

        return self.backbone(sp, coords_sp)
