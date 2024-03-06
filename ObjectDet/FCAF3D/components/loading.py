"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT License.
"""
# pylint: disable=no-member
from typing import Dict

import torch
import numpy as np

from mmdet3d.core.points import BasePoints, get_points_type
from mmdet3d.datasets.builder import PIPELINES
from mmdet3d.datasets.pipelines import LoadPointsFromFile, PointSegClassMapping
from mmdet.datasets.pipelines import LoadAnnotations
from mmdet3d.core.bbox.structures import BaseInstance3DBoxes, depth_box3d
import mmcv

@PIPELINES.register_module()
class BoxSegClassMapping(object):
    """
    Mapping raw label to target label for box annotations
    """
    def __init__(self) -> None:
        pass

@PIPELINES.register_module()
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
        pts_xyz = results['points'].tensor[..., :3].numpy()
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

        results['points'] = results.get('points')[uni_pts_idx]
        pts_instance_mask = results.get('pts_instance_mask', None)
        pts_semantic_mask = results.get('pts_semantic_mask', None)
        if pts_instance_mask is not None:
            pts_instance_mask = pts_instance_mask[uni_pts_idx]
            results['pts_instance_mask'] = pts_instance_mask
        if pts_semantic_mask is not None:
            pts_semantic_mask = pts_semantic_mask[uni_pts_idx]
            results['pts_semantic_mask'] = pts_semantic_mask
        return results


@PIPELINES.register_module()
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
        seg_field_name = 'pts_seg_fields'
        assert len(results[seg_field_name]) == 1

        points_xyz = results['points'].tensor[..., :3]
        
        if len(points_xyz) > self.num_points:
            if self.valid_sample:
                pts_seg = results[results[seg_field_name][0]]
                valid_indices = np.argwhere(pts_seg < self.num_classes)[..., 0]
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
        box_field_name = 'bbox3d_fields'
        seg_field_name = 'pts_seg_fields'
        gt_label_3d_name = 'gt_labels_3d'
        # Ignore empty annotated scenes
        if box_field_name not in results or not results[box_field_name]:
            return choices

        # Only support single type box annotations
        assert len(results[box_field_name]) == 1
        assert len(results[seg_field_name]) == 1

        pts_xyz = results['points'].tensor[..., :3].numpy()
        selected_pts_xyz = pts_xyz[choices]

        pts_seg = results[results[seg_field_name][0]]
        selected_pts_seg = pts_seg[choices]

        boxes_add_pts_idx = list()
        boxes_del_pts_idx = list()
        boxes_label = list()
        for b_key in results[box_field_name]:
            boxes: BaseInstance3DBoxes = results[b_key]
            boxes_corners = boxes.corners.numpy()
            boxes_tensor = list()
            boxes_ious = list()
            for b_idx, b_corners in enumerate(boxes_corners):
                b_label = results[gt_label_3d_name][b_idx]
                b_min, b_max = b_corners[0], b_corners[6]
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
                    boxes_tensor.append(boxes[b_idx].tensor)
                    boxes_label.append(b_label)
                    if b_p_iou < 0.95:
                        b_add_pts_idx = self._select_pts_idx_in_box_label(
                            pts_xyz, pts_seg, b_max, b_min, b_label)
                        boxes_add_pts_idx.extend(b_add_pts_idx)
                else:
                    boxes_del_pts_idx.extend(b_pts_idx)
            if not boxes_tensor:
                boxes_tensor = np.zeros((0, 6), dtype=np.float32)
                boxes_label = np.zeros((0,), dtype=np.int64)
            else:
                boxes_tensor = torch.concat(boxes_tensor, axis=0)
                boxes_label = np.array(boxes_label, dtype=np.int64)
            filtered_boxes = depth_box3d.DepthInstance3DBoxes(
                boxes_tensor, boxes_tensor.shape[-1], with_yaw=False)
            results[b_key] = filtered_boxes
            results[gt_label_3d_name] = boxes_label

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
        pts_instance_mask = results.get('pts_instance_mask', None)
        pts_semantic_mask = results.get('pts_semantic_mask', None)
        if pts_instance_mask is not None:
            pts_instance_mask = pts_instance_mask[choices]
            results['pts_instance_mask'] = pts_instance_mask
        if pts_semantic_mask is not None:
            pts_semantic_mask = pts_semantic_mask[choices]
            results['pts_semantic_mask'] = pts_semantic_mask
        return results


@PIPELINES.register_module()
class LoadPointsFromFileNormal(LoadPointsFromFile):
    """Load Points From File.

    Load points from file.

    Args:
        coord_type (str): The type of coordinates of points cloud.
            Available options includes:
            - 'LIDAR': Points in LiDAR coordinates.
            - 'DEPTH': Points in depth coordinates, usually for indoor dataset.
            - 'CAMERA': Points in camera coordinates.
        load_dim (int, optional): The dimension of the loaded points.
            Defaults to 6.
        use_dim (list[int], optional): Which dimensions of the points to use.
            Defaults to [0, 1, 2]. For KITTI dataset, set use_dim=4
            or use_dim=[0, 1, 2, 3] to use the intensity dimension.
        shift_height (bool, optional): Whether to use shifted height.
            Defaults to False.
        use_color (bool, optional): Whether to use color features.
            Defaults to False.
        file_client_args (dict, optional): Config dict of file clients,
            refer to
            https://github.com/open-mmlab/mmcv/blob/master/mmcv/fileio/file_client.py
            for more details. Defaults to dict(backend='disk').
    """

    def __init__(self,
                 coord_type,
                 load_dim=9,
                 use_dim=[0, 1, 2],
                 shift_height=False,
                 use_color=False,
                 use_normal=True,
                 file_client_args=dict(backend='disk')):
        self.shift_height = shift_height
        self.use_color = use_color
        self.use_normal = use_normal
        if isinstance(use_dim, int):
            use_dim = list(range(use_dim))
        assert max(use_dim) < load_dim, \
            f'Expect all used dimensions < {load_dim}, got {use_dim}'
        assert coord_type in ['CAMERA', 'LIDAR', 'DEPTH']

        self.coord_type = coord_type
        self.load_dim = load_dim
        self.use_dim = use_dim
        self.file_client_args = file_client_args.copy()
        self.file_client = None

    def _load_points(self, pts_filename):
        """Private function to load point clouds data.

        Args:
            pts_filename (str): Filename of point clouds data.

        Returns:
            np.ndarray: An array containing point clouds data.
        """
        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)
        try:
            pts_bytes = self.file_client.get(pts_filename)
            points = np.frombuffer(pts_bytes, dtype=np.float32)
        except ConnectionError:
            mmcv.check_file_exist(pts_filename)
            if pts_filename.endswith('.npy'):
                points = np.load(pts_filename)
            else:
                points = np.fromfile(pts_filename, dtype=np.float32)

        return points

    def __call__(self, results):
        """Call function to load points data from file.

        Args:
            results (dict): Result dict containing point clouds data.

        Returns:
            dict: The result dict containing the point clouds data.
                Added key and value are described below.

                - points (:obj:`BasePoints`): Point clouds data.
        """
        pts_filename = results['pts_filename']
        pts_filename = pts_filename.replace('points', 'points_normal')
        points = self._load_points(pts_filename)
        points = points.reshape(-1, self.load_dim)
        points = points[:, self.use_dim]
        attribute_dims = None

        if self.shift_height:
            floor_height = np.percentile(points[:, 2], 0.99)
            height = points[:, 2] - floor_height
            points = np.concatenate(
                [points[:, :3],
                 np.expand_dims(height, 1), points[:, 3:]], 1)
            attribute_dims = dict(height=3)

        if self.use_color:
            assert len(self.use_dim) >= 6
            if attribute_dims is None:
                attribute_dims = dict()
            attribute_dims.update(
                dict(color=[3,4,5]))
    
        if self.use_normal:
            assert len(self.use_dim) >= 6
            if attribute_dims is None:
                attribute_dims = dict()
            attribute_dims.update(
                dict(normal=[
                    points.shape[1] - 3,
                    points.shape[1] - 2,
                    points.shape[1] - 1,
                ]))

        points_class = get_points_type(self.coord_type)
        points = points_class(
            points, points_dim=points.shape[-1], attribute_dims=attribute_dims)
        results['points'] = points

        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__ + '('
        repr_str += f'shift_height={self.shift_height}, '
        repr_str += f'use_color={self.use_color}, '
        repr_str += f'use_color={self.use_normal}, '
        repr_str += f'file_client_args={self.file_client_args}, '
        repr_str += f'load_dim={self.load_dim}, '
        repr_str += f'use_dim={self.use_dim})'
        return repr_str

@PIPELINES.register_module()
class IntervalPrint(object):
    def __init__(self) -> None:
        pass
    def __call__(self, results: Dict):
        print("IntervalPrint Called")
        print(results.keys())
        print(results)
        return results


@PIPELINES.register_module()
class MyPointSegClassMapping(PointSegClassMapping):
    """Map original semantic class to valid category ids.

    Map valid classes as 0~len(valid_cat_ids)-1 and
    others as len(valid_cat_ids).

    Args:
        valid_cat_ids (tuple[int]): A tuple of valid category.
        max_cat_id (int, optional): The max possible cat_id in input
            segmentation mask. Defaults to 40.
    """

    def __init__(self, valid_cat_ids, max_cat_id=40):
        assert max_cat_id >= np.max(valid_cat_ids), \
            'max_cat_id should be greater than maximum id in valid_cat_ids'

        self.valid_cat_ids = valid_cat_ids
        self.max_cat_id = int(max_cat_id)

        # build cat_id to class index mapping
        neg_cls = len(valid_cat_ids)
        self.cat_id2class = np.ones(
            self.max_cat_id + 1, dtype=np.int) * neg_cls
        for cls_idx, cat_id in enumerate(valid_cat_ids):
            self.cat_id2class[cat_id] = cls_idx

    def __call__(self, results):
        """Call function to map original semantic class to valid category ids.

        Args:
            results (dict): Result dict containing point semantic masks.

        Returns:
            dict: The result dict containing the mapped category ids.
                Updated key and value are described below.

                - pts_semantic_mask (np.ndarray): Mapped semantic masks.
        """
        assert 'pts_semantic_mask' in results
        pts_semantic_mask = results['pts_semantic_mask']

        converted_pts_sem_mask = self.cat_id2class[pts_semantic_mask]
        results['pts_semantic_mask'] = converted_pts_sem_mask
        if converted_pts_sem_mask.shape[0] == 117667:
            raise NotImplementedError
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(valid_cat_ids={self.valid_cat_ids}, '
        repr_str += f'max_cat_id={self.max_cat_id})'
        return repr_str



@PIPELINES.register_module()
class MyLoadAnnotations3D(LoadAnnotations):
    """Load Annotations3D.

    Load instance mask and semantic mask of points and
    encapsulate the items into related fields.

    Args:
        with_bbox_3d (bool, optional): Whether to load 3D boxes.
            Defaults to True.
        with_label_3d (bool, optional): Whether to load 3D labels.
            Defaults to True.
        with_attr_label (bool, optional): Whether to load attribute label.
            Defaults to False.
        with_mask_3d (bool, optional): Whether to load 3D instance masks.
            for points. Defaults to False.
        with_seg_3d (bool, optional): Whether to load 3D semantic masks.
            for points. Defaults to False.
        with_bbox (bool, optional): Whether to load 2D boxes.
            Defaults to False.
        with_label (bool, optional): Whether to load 2D labels.
            Defaults to False.
        with_mask (bool, optional): Whether to load 2D instance masks.
            Defaults to False.
        with_seg (bool, optional): Whether to load 2D semantic masks.
            Defaults to False.
        with_bbox_depth (bool, optional): Whether to load 2.5D boxes.
            Defaults to False.
        poly2mask (bool, optional): Whether to convert polygon annotations
            to bitmasks. Defaults to True.
        seg_3d_dtype (dtype, optional): Dtype of 3D semantic masks.
            Defaults to int64
        file_client_args (dict): Config dict of file clients, refer to
            https://github.com/open-mmlab/mmcv/blob/master/mmcv/fileio/file_client.py
            for more details.
    """

    def __init__(self,
                 with_bbox_3d=True,
                 with_label_3d=True,
                 with_attr_label=False,
                 with_mask_3d=False,
                 with_seg_3d=False,
                 with_bbox=False,
                 with_label=False,
                 with_mask=False,
                 with_seg=False,
                 with_bbox_depth=False,
                 poly2mask=True,
                 seg_3d_dtype=np.int64,
                 file_client_args=dict(backend='disk')):
        super().__init__(
            with_bbox,
            with_label,
            with_mask,
            with_seg,
            poly2mask,
            file_client_args=file_client_args)
        self.with_bbox_3d = with_bbox_3d
        self.with_bbox_depth = with_bbox_depth
        self.with_label_3d = with_label_3d
        self.with_attr_label = with_attr_label
        self.with_mask_3d = with_mask_3d
        self.with_seg_3d = with_seg_3d
        self.seg_3d_dtype = seg_3d_dtype

    def _load_bboxes_3d(self, results):
        """Private function to load 3D bounding box annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing loaded 3D bounding box annotations.
        """
        results['gt_bboxes_3d'] = results['ann_info']['gt_bboxes_3d']
        results['bbox3d_fields'].append('gt_bboxes_3d')
        return results

    def _load_bboxes_depth(self, results):
        """Private function to load 2.5D bounding box annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing loaded 2.5D bounding box annotations.
        """
        results['centers2d'] = results['ann_info']['centers2d']
        results['depths'] = results['ann_info']['depths']
        return results

    def _load_labels_3d(self, results):
        """Private function to load label annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing loaded label annotations.
        """
        results['gt_labels_3d'] = results['ann_info']['gt_labels_3d']
        return results

    def _load_attr_labels(self, results):
        """Private function to load label annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing loaded label annotations.
        """
        results['attr_labels'] = results['ann_info']['attr_labels']
        return results

    def _load_masks_3d(self, results):
        """Private function to load 3D mask annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing loaded 3D mask annotations.
        """
        pts_instance_mask_path = results['ann_info']['pts_instance_mask_path']

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)
        try:
            mask_bytes = self.file_client.get(pts_instance_mask_path)
            pts_instance_mask = np.frombuffer(mask_bytes, dtype=np.int64)
        except ConnectionError:
            mmcv.check_file_exist(pts_instance_mask_path)
            pts_instance_mask = np.fromfile(
                pts_instance_mask_path, dtype=np.int64)

        results['pts_instance_mask'] = pts_instance_mask
        results['pts_mask_fields'].append('pts_instance_mask')
        return results

    def _load_semantic_seg_3d(self, results):
        """Private function to load 3D semantic segmentation annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing the semantic segmentation annotations.
        """
        pts_semantic_mask_path = results['ann_info']['pts_semantic_mask_path']
        print(pts_semantic_mask_path)
        if 'scene0154_00' in pts_semantic_mask_path:
            raise NotImplementedError
        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)
        try:
            mask_bytes = self.file_client.get(pts_semantic_mask_path)
            # add .copy() to fix read-only bug
            pts_semantic_mask = np.frombuffer(
                mask_bytes, dtype=self.seg_3d_dtype).copy()
        except ConnectionError:
            mmcv.check_file_exist(pts_semantic_mask_path)
            pts_semantic_mask = np.fromfile(
                pts_semantic_mask_path, dtype=np.int64)

        results['pts_semantic_mask'] = pts_semantic_mask
        results['pts_seg_fields'].append('pts_semantic_mask')
        return results

    def __call__(self, results):
        """Call function to load multiple types annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing loaded 3D bounding box, label, mask and
                semantic segmentation annotations.
        """
        results = super().__call__(results)
        if self.with_bbox_3d:
            results = self._load_bboxes_3d(results)
            if results is None:
                return None
        if self.with_bbox_depth:
            results = self._load_bboxes_depth(results)
            if results is None:
                return None
        if self.with_label_3d:
            results = self._load_labels_3d(results)
        if self.with_attr_label:
            results = self._load_attr_labels(results)
        if self.with_mask_3d:
            results = self._load_masks_3d(results)
        if self.with_seg_3d:
            results = self._load_semantic_seg_3d(results)

        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        indent_str = '    '
        repr_str = self.__class__.__name__ + '(\n'
        repr_str += f'{indent_str}with_bbox_3d={self.with_bbox_3d}, '
        repr_str += f'{indent_str}with_label_3d={self.with_label_3d}, '
        repr_str += f'{indent_str}with_attr_label={self.with_attr_label}, '
        repr_str += f'{indent_str}with_mask_3d={self.with_mask_3d}, '
        repr_str += f'{indent_str}with_seg_3d={self.with_seg_3d}, '
        repr_str += f'{indent_str}with_bbox={self.with_bbox}, '
        repr_str += f'{indent_str}with_label={self.with_label}, '
        repr_str += f'{indent_str}with_mask={self.with_mask}, '
        repr_str += f'{indent_str}with_seg={self.with_seg}, '
        repr_str += f'{indent_str}with_bbox_depth={self.with_bbox_depth}, '
        repr_str += f'{indent_str}poly2mask={self.poly2mask})'
        return repr_str
