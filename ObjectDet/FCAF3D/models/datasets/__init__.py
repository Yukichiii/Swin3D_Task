"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT License.
"""
from .s3dis_dataset import MyS3DISDataset
from .scannet_dataset import MyScanNetDataset

__all__ = [
    'MyS3DISDataset',
    'MyScanNetDataset',
]
