import numpy as np
import random

import torch

from util.voxelize import voxelize, voxelize_and_inverse


class RandomDropout(object):
    def __init__(self, dropout_ratio=0.2, dropout_application_ratio=0.5):
        """
        upright_axis: axis index among x,y,z, i.e. 2 for z
        """
        self.dropout_ratio = dropout_ratio
        self.dropout_application_ratio = dropout_application_ratio

    def __call__(self, coords, feats, labels):
        if random.random() < self.dropout_ratio:
            N = len(coords)
            inds = np.random.choice(N, int(N * (1 - self.dropout_ratio)), replace=False)
            return coords[inds], feats[inds], labels[inds]
        return coords, feats, labels


def collate_fn(batch):
    coord, feat, label = list(zip(*batch))
    offset, count = [], 0
    # print("coord:", len(coord))
    for item in coord:
        # print("item shape:",item.shape)
        count += item.shape[0]
        offset.append(count)
    return torch.cat(coord), torch.cat(feat), torch.cat(label), torch.IntTensor(offset)


def collate_fn_pts(batch):
    coord, feat, label, label_pts, inverse_map = list(zip(*batch))
    offset, count = [], 0
    # print("coord:", len(coord))
    inverse_list = []
    for item, inverse in zip(coord, inverse_map):
        # print("item shape:",item.shape)
        inverse_list.append(inverse + count)
        count += item.shape[0]
        offset.append(count)
    return (
        torch.cat(coord),
        torch.cat(feat),
        torch.cat(label),
        torch.IntTensor(offset),
        torch.cat(label_pts),
        torch.cat(inverse_list),
    )


def data_prepare_s3dis(
    coord,
    feat,
    label,
    split="train",
    voxel_size=0.04,
    voxel_max=None,
    transform=None,
    shuffle_index=False,
):
    if transform:
        # coord, feat, label = transform(coord, feat, label)
        color = feat[:, 0:3]
        coord, color = transform(coord, color)
        feat[:, 0:3] = color
        # if split=='train':
        #     coord, feat, label = RandomDropout(0.2)(coord, feat, label)
    if voxel_size:
        coord_min = np.min(coord, 0)
        coord -= coord_min
        coord = coord.astype(np.float32)
        uniq_idx = voxelize(coord, voxel_size)
        coord, feat, label = coord[uniq_idx], feat[uniq_idx], label[uniq_idx]
        coord = coord / voxel_size
    if voxel_max and label.shape[0] > voxel_max:
        init_idx = (
            np.random.randint(label.shape[0])
            if "train" in split
            else label.shape[0] // 2
        )
        crop_idx = np.argsort(np.sum(np.square(coord - coord[init_idx]), 1))[:voxel_max]
        coord, feat, label = coord[crop_idx], feat[crop_idx], label[crop_idx]
    if shuffle_index:
        shuf_idx = np.arange(coord.shape[0])
        np.random.shuffle(shuf_idx)
        coord, feat, label = coord[shuf_idx], feat[shuf_idx], label[shuf_idx]

    # coord_min = np.min(coord, 0)
    # coord -= coord_min
    coord = torch.FloatTensor(coord)
    feat = torch.FloatTensor(feat)
    label = torch.LongTensor(label)
    return coord, feat, label


def data_prepare_s3dis_point(
    coord,
    feat,
    label,
    split="train",
    voxel_size=0.04,
    voxel_max=None,
    transform=None,
    shuffle_index=False,
):
    if transform:
        # coord, feat, label = transform(coord, feat, label)
        color = feat[:, 0:3]
        coord, color = transform(coord, color)
        feat[:, 0:3] = color
    label_pts = label
    if voxel_size:
        coord_min = np.min(coord, 0)
        coord -= coord_min
        coord = coord.astype(np.float32)
        coord = coord / voxel_size
        int_coord = coord.astype(np.int32)

        unique_map, inverse_map = voxelize_and_inverse(int_coord, voxel_size)
        # print(len(unique_map), len(inverse_map))
        coord = coord[unique_map]
        feat = feat[unique_map]
        label = label[unique_map]

    # coord_min = np.min(coord, 0)
    # coord -= coord_min
    coord = torch.FloatTensor(coord)
    feat = torch.FloatTensor(feat)
    label_pts = torch.LongTensor(label_pts)
    label = torch.LongTensor(label)
    inverse_map = torch.LongTensor(inverse_map)
    return coord, feat, label, label_pts, inverse_map


def data_prepare_scannet_point(
    coord,
    feat,
    label,
    split="train",
    voxel_size=0.04,
    voxel_max=None,
    transform=None,
    shuffle_index=False,
):
    if transform:
        # coord, feat, label = transform(coord, feat, label)
        color = feat[:, 0:3]
        normal = feat[:, 3:6]
        if normal.shape[1] == 0:
            normal = None
            coord, color = transform(coord, color)
        else:
            coord, color, normal = transform(coord, color, normal)
            feat[:, 3:6] = normal
        feat[:, 0:3] = color
    label_pts = label
    if voxel_size:
        coord_min = np.min(coord, 0)
        coord -= coord_min
        coord = coord.astype(np.float32)
        coord = coord / voxel_size
        int_coord = coord.astype(np.int32)

        unique_map, inverse_map = voxelize_and_inverse(int_coord, voxel_size)

        coord = coord[unique_map]
        feat = feat[unique_map]
        label = label[unique_map]

    # coord_min = np.min(coord, 0)
    # coord -= coord_min
    coord = torch.FloatTensor(coord)
    feat = torch.FloatTensor(feat)
    label_pts = torch.LongTensor(label_pts)
    label = torch.LongTensor(label)
    inverse_map = torch.LongTensor(inverse_map)
    return coord, feat, label, label_pts, inverse_map


def data_prepare_scannet(
    coord,
    feat,
    label,
    split="train",
    voxel_size=0.04,
    voxel_max=None,
    transform=None,
    shuffle_index=False,
):
    if transform:
        # coord, feat, label = transform(coord, feat, label)
        color = feat[:, 0:3]
        normal = feat[:, 3:6]
        if normal.shape[1] == 0:
            normal = None
            coord, color = transform(coord, color)
        else:
            coord, color, normal = transform(coord, color, normal)
            feat[:, 3:6] = normal
        feat[:, 0:3] = color
        # if split=='train':
        #     coord, feat, label = RandomDropout(0.2)(coord, feat, label)
    if voxel_size:
        coord_min = np.min(coord, 0)
        coord -= coord_min
        coord = coord.astype(np.float32)
        uniq_idx = voxelize(coord, voxel_size)
        coord, feat, label = coord[uniq_idx], feat[uniq_idx], label[uniq_idx]
        coord = coord / voxel_size
    if voxel_max and label.shape[0] > voxel_max:
        init_idx = (
            np.random.randint(label.shape[0])
            if "train" in split
            else label.shape[0] // 2
        )
        crop_idx = np.argsort(np.sum(np.square(coord - coord[init_idx]), 1))[:voxel_max]
        coord, feat, label = coord[crop_idx], feat[crop_idx], label[crop_idx]
    if shuffle_index:
        shuf_idx = np.arange(coord.shape[0])
        np.random.shuffle(shuf_idx)
        coord, feat, label = coord[shuf_idx], feat[shuf_idx], label[shuf_idx]

    # coord_min = np.min(coord, 0)
    # coord -= coord_min
    coord = torch.FloatTensor(coord)
    feat = torch.FloatTensor(feat)
    label = torch.LongTensor(label)
    return coord, feat, label
