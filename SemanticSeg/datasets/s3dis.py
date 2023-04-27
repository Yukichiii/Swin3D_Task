import os
import numpy as np
import torch
from torch.utils.data import Dataset

from util.data_util import data_prepare_s3dis as data_prepare
from util.data_util import data_prepare_s3dis_point


class S3DIS(Dataset):
    def __init__(
        self,
        split="train",
        data_root="trainval",
        test_area=5,
        voxel_size=0.04,
        voxel_max=None,
        transform=None,
        shuffle_index=False,
        loop=1,
        vote_num=1,
    ):
        super().__init__()
        (
            self.split,
            self.voxel_size,
            self.transform,
            self.voxel_max,
            self.shuffle_index,
            self.loop,
        ) = (split, voxel_size, transform, voxel_max, shuffle_index, loop)
        data_list = sorted(os.listdir(data_root))
        data_list = [item[:-4] for item in data_list if "Area_" in item]
        if split == "train":
            self.data_list = [
                item for item in data_list if not "Area_{}".format(test_area) in item
            ]
        else:
            self.data_list = [
                item for item in data_list if "Area_{}".format(test_area) in item
            ]
        self.data_root = data_root
        # for item in self.data_list:
        #     if not os.path.exists("/dev/shm/{}".format(item)):
        #         data_path = os.path.join(data_root, item + '.npy')
        #         data = np.load(data_path)  # xyzrgbl, N*7
        #         sa_create("shm://{}".format(item), data)
        self.data_idx = np.arange(len(self.data_list))
        print("Totally {} samples in {} set.".format(len(self.data_idx), split))

        self.data_idx = np.repeat(self.data_idx, vote_num)
        print("Total repeated {} samples in {} set.".format(len(self.data_idx), split))

    def __getitem__(self, idx):
        data_idx = self.data_idx[idx % len(self.data_idx)]

        # data = SA.attach("shm://{}".format(self.data_list[data_idx])).copy()
        item = self.data_list[data_idx]
        data_path = os.path.join(self.data_root, item + ".npy")
        data = np.load(data_path)

        coord, feat, label = data[:, 0:3], data[:, 3:6], data[:, 6]
        feat = feat / 127.5 - 1
        coord, feat, label = data_prepare(
            coord,
            feat,
            label,
            self.split,
            self.voxel_size,
            self.voxel_max,
            self.transform,
            self.shuffle_index,
        )
        return coord, feat, label

    def __len__(self):
        return len(self.data_idx) * self.loop


class S3DIS_Point(S3DIS):
    def __getitem__(self, idx):
        data_idx = self.data_idx[idx % len(self.data_idx)]

        # data = SA.attach("shm://{}".format(self.data_list[data_idx])).copy()
        item = self.data_list[data_idx]
        data_path = os.path.join(self.data_root, item + ".npy")
        data = np.load(data_path)

        coord, feat, label = data[:, 0:3], data[:, 3:6], data[:, 6]
        feat = feat / 127.5 - 1
        coord, feat, label, label_pts, inverse_map = data_prepare_s3dis_point(
            coord,
            feat,
            label,
            self.split,
            self.voxel_size,
            self.voxel_max,
            self.transform,
            self.shuffle_index,
        )
        return coord, feat, label, label_pts, inverse_map
