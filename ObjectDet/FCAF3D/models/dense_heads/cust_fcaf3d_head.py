"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT License.
"""
# pylint: disable=abstract-method,unbalanced-tuple-unpacking,invalid-name

import torch
from mmdet3d.models.dense_heads.fcaf3d_head import FCAF3DHead
from mmdet3d.models import HEADS
import MinkowskiEngine as ME
from MinkowskiEngine import SparseTensor

@HEADS.register_module()
class CustomFCAF3DHead(FCAF3DHead):
    def _prune(self, x, scores):
        """Prunes the tensor by score thresholding.

        Args:
            x (SparseTensor): Tensor to be pruned.
            scores (SparseTensor): Scores for thresholding.

        Returns:
            SparseTensor: Pruned tensor.
        """
        with torch.no_grad():
            scores = SparseTensor(scores.F.float(), coordinate_map_key=scores.coordinate_map_key, coordinate_manager=scores.coordinate_manager)
            coordinates = x.C.float()
            interpolated_scores = scores.features_at_coordinates(coordinates)
            prune_mask = interpolated_scores.new_zeros(
                (len(interpolated_scores)), dtype=torch.bool)
            for permutation in x.decomposition_permutations:
                score = interpolated_scores[permutation]
                mask = score.new_zeros((len(score)), dtype=torch.bool)
                topk = min(len(score), self.pts_prune_threshold)
                ids = torch.topk(score.squeeze(1), topk, sorted=False).indices
                mask[ids] = True
                prune_mask[permutation[mask]] = True
        x = self.pruning(x, prune_mask)
        return x
