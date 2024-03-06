"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT License.
"""
import os
from os import path as osp

import numpy as np

from mmdet3d.datasets import DATASETS, ScanNetDataset
from .show_result import show_result


@DATASETS.register_module()
class MyScanNetDataset(ScanNetDataset):
    def show(self, results, out_dir, show=True, pipeline=None):
        """Results visualization.

        Args:
            results (list[dict]): List of bounding boxes results.
            out_dir (str): Output directory of visualization result.
            show (bool): Visualize the results online.
            pipeline (list[dict], optional): raw data loading for showing.
                Default: None.
        """
        self.test_mode = False
        assert out_dir is not None, 'Expect out_dir, got none.'
        if not osp.exists(out_dir):
            os.makedirs(out_dir)
        pipeline = self._get_pipeline(pipeline)
        for i, result in enumerate(results):
            data_info = self.data_infos[i]
            pts_path = data_info['pts_path']
            file_name = osp.split(pts_path)[-1].split('.')[0]
            points = self._extract_data(i, pipeline, 'points').numpy()
            gt_bboxes = self.get_ann_info(i)['gt_bboxes_3d'].tensor.numpy()
            gt_labels = self.get_ann_info(i)['gt_labels_3d']
            
            score_threshold = 0.2
            pred_bboxes = result['boxes_3d']
            pred_labels = result['labels_3d']
            pred_scores = result['scores_3d']
            if not isinstance(pred_bboxes, np.ndarray):
                pred_bboxes = pred_bboxes.tensor.numpy()
                pred_labels = pred_labels.numpy()
                pred_scores = pred_scores.numpy()

            score_threshold = 0.2
            pred_mask = pred_scores > score_threshold
            pred_bboxes = pred_bboxes[pred_mask]
            pred_labels = pred_labels[pred_mask]
            pred_scores = pred_scores[pred_mask]

            print(points.shape, gt_bboxes.shape, gt_labels.shape, pred_bboxes.shape, pred_labels.shape)
            np.savez(out_dir+"/%s.npz"%file_name, points=points, gt_bboxes=gt_bboxes, gt_labels=gt_labels, pred_bboxes=pred_bboxes, pred_labels=pred_labels)
            # np.save(out_dir+"/%s_gt.npy"%file_name, gt_bboxes)
            # np.save(out_dir+"/%s_pred.npy"%file_name, pred_bboxes)
            # np.save(out_dir+"/%s_pts.npy"%file_name, points)
            # show_result(points, gt_bboxes, pred_bboxes, out_dir, file_name,
            #             show, snapshot=True)