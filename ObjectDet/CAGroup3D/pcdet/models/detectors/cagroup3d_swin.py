
"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT License.
"""
from .cagroup3d import CAGroup3D
import torch
import MinkowskiEngine as ME

class CAGroup3D_Swin(CAGroup3D):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()
        # set hparams
        self.voxel_size = self.model_cfg.VOXEL_SIZE
        self.semantic_min_threshold = self.model_cfg.SEMANTIC_MIN_THR
        self.semantic_iter_value = self.model_cfg.SEMANTIC_ITER_VALUE
        self.semantic_value = self.model_cfg.SEMANTIC_THR
    
    def voxelization(self, points):
        """voxelize input points."""
        # points Nx7 (bs_id, x, y, z, r, g, b)
        coordinates = points[:, :4].clone()
        coordinates[:, 1:] /= self.voxel_size
        features = points[:, 4:].clone()
        voxel_offset = coordinates[:, 1:].clone()
        voxel_offset = voxel_offset - voxel_offset .int()
        features = torch.cat([voxel_offset, features], dim=1)
        sp_tensor = ME.SparseTensor(coordinates=coordinates, features=features)
        return sp_tensor

    def forward(self, batch_dict):
        # adjust semantic value
        cur_epoch = batch_dict.get('cur_epoch', None)
        assert cur_epoch is not None
        self.module_list[1].semantic_threshold = max(self.semantic_value - int(cur_epoch) * self.semantic_iter_value, self.semantic_min_threshold)
        # normalize point features
        batch_dict['points'][:, -3:] = batch_dict['points'][:, -3:] / 255.
        sp_tensor = self.voxelization(batch_dict['points'])
        batch_dict['sp_tensor'] = sp_tensor
        
        for cur_module in self.module_list:
            results = cur_module(batch_dict)
            batch_dict.update(results)

        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss(batch_dict)
            ret_dict = {
                'loss': loss
            }
            disp_dict['cur_semantic_value'] = self.module_list[1].semantic_threshold
            return ret_dict, tb_dict, disp_dict
        else:
            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            return pred_dicts, recall_dicts
