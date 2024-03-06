"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT License.
"""
# pylint: disable=abstract-method,unbalanced-tuple-unpacking,invalid-name
try:
    import MinkowskiEngine as ME
except ImportError:
    # Please follow getting_started.md to install MinkowskiEngine.
    pass

from mmdet3d.models.detectors.mink_single_stage import MinkSingleStage3DDetector
from mmdet3d.models import DETECTORS
import torch

@DETECTORS.register_module()
class CustomMinkSingleStage3DDetector(MinkSingleStage3DDetector):
    """
    Customized Mink-single stage 3D Detector
    """
    def __init__(self, backbone, head, voxel_size, train_cfg=None,
        test_cfg=None, init_cfg=None, pretrained=None, use_xyz=False):
        super().__init__(backbone, head, voxel_size, train_cfg, test_cfg, \
            init_cfg, pretrained)
        self.use_xyz = use_xyz

    def extract_feat(self, points):
        """
        Returns:
            SparseTensor: Voxelized point clouds.
        """
        if self.use_xyz:
            # input points: xyz, rgb, (normal)
            # ->
            # features: rgb, (normal), xyz
            coordinates, features = ME.utils.batch_sparse_collate(
                [(p[:, :3] / self.voxel_size, torch.cat([p[:, 3:], p[:, :3]], dim=1)) for p in points],
                device=points[0].device)
        else:
            coordinates, features = ME.utils.batch_sparse_collate(
                [(p[:, :3] / self.voxel_size, p[:, 3:]) for p in points],
                device=points[0].device)
        x = ME.SparseTensor(coordinates=coordinates, features=features)
        x = self.backbone(x)
        return x

@DETECTORS.register_module()
class CustomMinkSingleStage3DDetector_v2(MinkSingleStage3DDetector):
    """
    Customized Mink-single stage 3D Detector
    """
    def __init__(self, backbone, head, voxel_size, train_cfg=None,
        test_cfg=None, init_cfg=None, pretrained=None, use_xyz=False):
        super().__init__(backbone, head, voxel_size, train_cfg, test_cfg, \
            init_cfg, pretrained)
        self.use_xyz = use_xyz

    def extract_feat(self, points):
        """
        Returns:
            SparseTensor: Voxelized point clouds.
        """
        if self.use_xyz:
            # input points: xyz, rgb, (normal)
            # ->
            # features: rgb, (normal), xyz
            coordinates, features = ME.utils.batch_sparse_collate(
                [((p[:, :3] / self.voxel_size), torch.cat([p[:, 3:], (p[:, :3] / self.voxel_size) - torch.floor(p[:, :3] / self.voxel_size)], dim=1)) for p in points],
                device=points[0].device)
        else:
            coordinates, features = ME.utils.batch_sparse_collate(
                [((p[:, :3] / self.voxel_size), p[:, 3:]) for p in points],
                device=points[0].device)
        x = ME.SparseTensor(coordinates=coordinates, features=features)
        x = self.backbone(x)
        return x
