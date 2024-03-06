"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT License.
"""
import numpy as np
from typing import Dict
import torch

class VoxelStridePointSampler(object):
    """
    Point sampler based on the voxel stride

    Args:
        voxel_stride (float): the voxel stride
    """
    def __init__(self, voxel_stride, is_random=False) -> None:
        self.voxel_stride = voxel_stride
        self.is_random = is_random

    @staticmethod
    def hash_vec_fnv_1a(in_vec: np.ndarray) -> np.ndarray:
        """
        Fowler-Noll-Vo-1a hash function
        """
        assert len(in_vec.shape) == 2
        prime = 1099511628211
        basis = 14695981039346656037

        hash_code = basis * np.ones(in_vec.shape[0], dtype=np.uint64)
        for d_idx in range(in_vec.shape[1]):
            hash_code = np.bitwise_xor(hash_code, in_vec[..., d_idx])
            hash_code *= prime
        return hash_code

    def __call__(self, results: Dict):
        pts_xyz = results['points'][..., :3].copy()
        if self.is_random:
            resort_idx = np.arange(pts_xyz.shape[0], dtype=np.int32)
            np.random.shuffle(resort_idx)
            pts_xyz = pts_xyz[resort_idx]

        pts_min = np.min(pts_xyz, axis=0, keepdims=True)
        pts_xyz -= pts_min

        pts_voxel = (pts_xyz / self.voxel_stride).astype(np.uint64)
        pts_hash = self.hash_vec_fnv_1a(pts_voxel)
        _, uni_pts_idx = np.unique(pts_hash, return_index=True)
        if self.is_random:
            uni_pts_idx = resort_idx[uni_pts_idx]

        results['points'] = results['points'][uni_pts_idx]
        instance_mask = results.get('instance_mask', None)
        semantic_mask = results.get('semantic_mask', None)
        if instance_mask is not None:
            instance_mask = instance_mask[uni_pts_idx]
            results['instance_mask'] = instance_mask
        if semantic_mask is not None:
            semantic_mask = semantic_mask[uni_pts_idx]
            results['semantic_mask'] = semantic_mask
        return results


class DistancePointSampler(object):
    """
    Sample point based on distance with the selected point

    Args:
        num_points (int): the number of points to be reserved
        box_threshold (float): the IoU threshold to reserve the box annotation.
            Defaults to 0.25.
        realignment (bool): whether to realign between the points and annotations.
            Defaults to False.
    """
    def __init__(self, num_points, box_threshold=0.5, valid_sample=False, \
        num_classes=40) -> None:
        self.num_points = num_points
        self.box_threshold = box_threshold
        self.valid_sample = valid_sample
        self.num_classes = num_classes

    def _points_random_nearest_sampling(self, results: Dict)\
        -> np.ndarray:
        points_xyz = results['points'][..., :3]

        if len(points_xyz) > self.num_points:
            if self.valid_sample:
                pts_seg = results["semantic_mask"]
                valid_indices = np.argwhere(pts_seg < self.num_classes)[..., 0]
                if len(valid_indices)==0:
                    print(results["frame_id"], pts_seg.shape, results["gt_names"], results["gt_boxes_mask"], results["gt_boxes_label"])
                    init_choice = 0
                else:
                    valid_choice = int(np.random.random() * len(valid_indices))
                    init_choice = valid_indices[valid_choice]
            else:
                init_choice = int(np.random.random() * len(points_xyz))
            init_points = points_xyz[init_choice]
            dist = np.linalg.norm(points_xyz - init_points, axis=-1)
            sort_dist = np.argsort(dist)
            choices = sort_dist[:self.num_points]
        else:
            choices = np.arange(len(points_xyz), dtype=np.int32)
        return choices

    @staticmethod
    def _select_pts_idx_in_box(pts_xyz: np.ndarray, box_max: np.ndarray,
        box_min: np.ndarray):
        return np.argwhere(np.all(pts_xyz <= box_max, axis=-1, keepdims=False)\
            & np.all(pts_xyz >= box_min, axis=-1, keepdims=False))[..., 0]

    @staticmethod
    def _select_pts_idx_in_box_label(pts_xyz: np.ndarray, pts_seg, box_max:
        np.ndarray, box_min: np.ndarray, seg_id: int) -> np.ndarray:
        t_pts_idx = __class__._select_pts_idx_in_box(pts_xyz, box_max, box_min)  # pylint: disable=protected-access
        t_pts_idx = t_pts_idx[np.argwhere(pts_seg[t_pts_idx] == seg_id)[..., 0]]
        return t_pts_idx

    def _box3d_iou_sampling(self, choices, results: Dict) -> np.ndarray:
        # Ignore empty annotated scenes
        pts_xyz = results['points'][..., :3]
        selected_pts_xyz = pts_xyz[choices]

        pts_seg = results["semantic_mask"]
        selected_pts_seg = pts_seg[choices]

        boxes_add_pts_idx = list()
        boxes_del_pts_idx = list()
        boxes_label = list()
        boxes_name = list()
        boxes_mask = list()

        boxes = results["gt_boxes"]

        boxes_tensor = list()
        boxes_ious = list()

        for b_idx, b_corners in enumerate(boxes):
            b_label = results["gt_boxes_label"][b_idx]
            b_name = results["gt_names"][b_idx]
            b_mask = results["gt_boxes_mask"][b_idx]

            box_i = boxes[b_idx]
            b_min = box_i[:3] - box_i[3:6]/2
            b_max = box_i[:3] + box_i[3:6]/2
            b_pts_idx = self._select_pts_idx_in_box_label(selected_pts_xyz, \
                selected_pts_seg, b_max, b_min, b_label)
                
            if b_pts_idx.size == 0:
                boxes_ious.append(0.)
                continue
            b_pts = selected_pts_xyz[b_pts_idx]
            b_pts_min = np.min(b_pts, axis=0)
            b_pts_max = np.max(b_pts, axis=0)

            b_p_joint = np.prod(b_pts_max - b_pts_min) + np.prod(b_max - b_min)
            b_p_intr = np.prod(np.maximum(np.minimum(b_pts_max, b_max) - \
                np.maximum(b_pts_min, b_min), 0))
            b_p_iou = b_p_intr / (b_p_joint - b_p_intr)
            boxes_ious.append(b_p_iou)

            if b_p_iou > self.box_threshold:
                boxes_tensor.append(boxes[b_idx])
                boxes_label.append(b_label)
                boxes_name.append(b_name)
                boxes_mask.append(b_mask)
                if b_p_iou < 0.95:
                    b_add_pts_idx = self._select_pts_idx_in_box_label(
                        pts_xyz, pts_seg, b_max, b_min, b_label)
                    boxes_add_pts_idx.extend(b_add_pts_idx)
            else:
                boxes_del_pts_idx.extend(b_pts_idx)
        if not boxes_tensor:
            boxes_tensor = np.zeros((0, 7), dtype=np.float32)
            boxes_label = np.zeros((0,), dtype=np.int64)
        else:
            boxes_tensor = np.stack(boxes_tensor, axis=0)
            boxes_label = np.array(boxes_label, dtype=np.int64)

        #print(boxes.shape, boxes_tensor.shape, boxes_ious)
        results["gt_boxes"] = boxes_tensor
        results["gt_boxes_label"] = boxes_label
        results["gt_names"] = np.array(boxes_name, dtype=str)
        results["gt_boxes_mask"] = np.array(boxes_mask, dtype=np.bool)

        if boxes_del_pts_idx:
            boxes_del_pts_idx = np.unique(boxes_del_pts_idx)
            choices = np.delete(choices, boxes_del_pts_idx, axis=0)

        if boxes_add_pts_idx:
            choices = np.concatenate([choices, boxes_add_pts_idx], axis=0)
            choices = np.unique(choices)

        if len(choices) > self.num_points:
            selected_pts_seg = pts_seg[choices]
            seg_pts_mask = np.isin(selected_pts_seg, boxes_label)
            valid_pts_idx= choices[seg_pts_mask]
            invalid_pts_idx = choices[~seg_pts_mask]
            np.random.shuffle(invalid_pts_idx)
            invalid_pts_idx = invalid_pts_idx[:self.num_points - len(valid_pts_idx)]
            choices = np.concatenate([valid_pts_idx, invalid_pts_idx], axis=0)
        return choices

    def __call__(self, results: Dict):
        # Step 1: Find sampled points
        choices = self._points_random_nearest_sampling(results)

        # Step 2: Find and update contained annotations
        choices = self._box3d_iou_sampling(choices, results)
        # Step 3: Update instance and semantic masks
        results['points'] = results.get('points')[choices]
        instance_mask = results.get('instance_mask', None)
        semantic_mask = results.get('semantic_mask', None)
        if instance_mask is not None:
            instance_mask = instance_mask[choices]
            results['instance_mask'] = instance_mask
        if semantic_mask is not None:
            semantic_mask = semantic_mask[choices]
            results['semantic_mask'] = semantic_mask
        return results
