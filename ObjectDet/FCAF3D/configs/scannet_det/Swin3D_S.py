"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT License.
"""
# pylint: disable=invalid-name
_base_ = ['../_base_/default_runtime.py']
n_points = 60000
voxel_stride = 0.02

dataset_type = 'ScanNetDataset'
data_root = '/mnt/code/mmdetection3d_yuxgu/data/scannet/'
class_names = ('cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window',
               'bookshelf', 'picture', 'counter', 'desk', 'curtain',
               'refrigerator', 'showercurtrain', 'toilet', 'sink', 'bathtub',
               'garbagebin')
valid_class_ids = (3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34,
                   36, 39)

train_pipeline = [
    dict(
        type='LoadPointsFromFileNormal',
        coord_type='DEPTH',
        shift_height=False,
        use_color=True,
        use_normal=False,
        load_dim=9,
        use_dim=[0, 1, 2, 3, 4, 5]),
    dict(type='LoadAnnotations3D', with_seg_3d=True),
    dict(
        type='PointSegClassMapping',
        valid_cat_ids=valid_class_ids,
        max_cat_id=40),
    dict(type='GlobalAlignment', rotation_axis=2),
    dict(
        type='VoxelStridePointSampler',
        voxel_stride=voxel_stride,
        is_random=True),
    dict(
        type='DistancePointSampler',
        valid_sample=True,
        #box_threshold=0.75,
        num_classes=len(valid_class_ids),
        num_points=n_points),
    # dict(type='PointSample', num_points=n_points),
    dict(
        type='RandomFlip3D',
        sync_2d=False,
        flip_ratio_bev_horizontal=0.5,
        flip_ratio_bev_vertical=0.5),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.087266, 0.087266],
        scale_ratio_range=[.9, 1.1],
        translation_std=[.1, .1, .1],
        shift_height=False),
    dict(type='NormalizePointsColor', color_mean=None),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
]
test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='DEPTH',
        shift_height=False,
        use_color=True,
        load_dim=6,
        use_dim=[0, 1, 2, 3, 4, 5]),
    dict(type='GlobalAlignment', rotation_axis=2),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='GlobalRotScaleTrans',
                rot_range=[0, 0],
                scale_ratio_range=[1., 1.],
                translation_std=[0, 0, 0]),
            dict(
                type='RandomFlip3D',
                sync_2d=False,
                flip_ratio_bev_horizontal=0.5,
                flip_ratio_bev_vertical=0.5),
            dict(
                type='VoxelStridePointSampler',
                voxel_stride=voxel_stride,
                is_random=True),
            # dict(type='PointSample', num_points=n_points),
            dict(type='NormalizePointsColor', color_mean=None),
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='Collect3D', keys=['points'])
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=4,
    train=dict(
        type='RepeatDataset',
        times=10,
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            ann_file=data_root + 'scannet_infos_train.pkl',
            pipeline=train_pipeline,
            filter_empty_gt=True,
            classes=class_names,
            box_type_3d='Depth')),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'scannet_infos_val.pkl',
        pipeline=test_pipeline,
        classes=class_names,
        test_mode=True,
        box_type_3d='Depth'),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'scannet_infos_val.pkl',
        pipeline=test_pipeline,
        classes=class_names,
        test_mode=True,
        box_type_3d='Depth'))
model = dict(
    type='CustomMinkSingleStage3DDetector_v2',
    use_xyz=True,
    voxel_size=.02,
    backbone=dict(
        type='Swin3DEncoder_RGB',
        in_channels=6,
        depths=[2,4,9,4,4],
        channels = [48, 96, 192, 384, 384],
        num_heads=[6, 6, 12, 24, 24],
        window_size=7,
        knn_down=True,
        stem_transformer=True),
    head=dict(
        type='FCAF3DHead',
        in_channels=(96, 192, 384, 384),
        out_channels=128,
        voxel_size=.02,
        pts_prune_threshold=100000,
        pts_assign_threshold=27,
        pts_center_threshold=18,
        n_classes=18,
        n_reg_outs=6),
    train_cfg=dict(),
    test_cfg=dict(nms_pre=1000, iou_thr=.5, score_thr=.01))

find_unused_parameters = True
optimizer = dict(type='AdamW', lr=0.001, weight_decay=0.0001,
    paramwise_cfg=dict(custom_keys={'.blocks': dict(lr_mult=0.1)}))
optimizer_config = dict(grad_clip=dict(max_norm=10, norm_type=2))
lr_config = dict(policy='step', warmup=None, step=[8, 11])
runner = dict(type='EpochBasedRunner', max_epochs=12)
checkpoint_config = dict(interval=1, create_symlink=False)
custom_hooks = [dict(type='EmptyCacheHook', after_iter=True)]
