import os
import numpy as np
import torch
from torch.utils.data import Dataset

from util.data_util import data_prepare_scannet as data_prepare
from util.data_util import data_prepare_scannet_point
import glob
import trimesh

from plyfile import PlyData, PlyElement
import pandas as pd


def read_plyfile(filepath):
    """Read ply file and return it as numpy array. Returns None if emtpy."""
    with open(filepath, "rb") as f:
        plydata = PlyData.read(f)
    if plydata.elements:
        return pd.DataFrame(plydata.elements[0].data).values


remapper = np.ones(150) * (-100)
for i, x in enumerate(
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39]
):
    remapper[x] = i


class Scannetv2(Dataset):
    def __init__(
        self,
        split="train",
        data_root="trainval",
        voxel_size=0.04,
        voxel_max=None,
        transform=None,
        shuffle_index=False,
        loop=1,
    ):
        super().__init__()

        self.split = split
        self.data_root = data_root
        self.voxel_size = voxel_size
        self.voxel_max = voxel_max
        self.transform = transform
        self.shuffle_index = shuffle_index
        self.loop = loop

        if split == "train" or split == "val":
            self.data_list = glob.glob(os.path.join(data_root, split, "*.pth"))
            self.data_list.sort()
        elif split == "trainval":
            self.data_list = glob.glob(
                os.path.join(data_root, "train", "*.pth")
            ) + glob.glob(os.path.join(data_root, "val", "*.pth"))
        elif split == "test":
            self.data_list = glob.glob(os.path.join(data_root, split, "*.pth"))
        elif split == "trainval_v2":
            train_data_list = glob.glob(os.path.join(data_root, "train", "*.pth"))
            val_data_list = glob.glob(os.path.join(data_root, "val", "*.pth"))
            train_data_list.sort()
            val_data_list.sort()
            self.data_list = train_data_list + val_data_list[:-100]
            print("not include in train: ", val_data_list[-100:])
        elif split == "val_v2":
            val_data_list = glob.glob(os.path.join(data_root, "val", "*.pth"))
            val_data_list.sort()
            print("val data: ", val_data_list[-100:])
            self.data_list = val_data_list[-100:]

        elif split == "val_v3":
            val_data_list = glob.glob(os.path.join(data_root, "val", "*.pth"))
            val_data_list.sort()
            print("val data: ", val_data_list[:-100])
            self.data_list = val_data_list[:-100]
        else:
            raise ValueError("no such split: {}".format(split))
        print("voxel_size: ", voxel_size)
        print("Totally {} samples in {} set.".format(len(self.data_list), split))

    def __getitem__(self, idx):
        # data_idx = self.data_idx[idx % len(self.data_idx)]

        # data = SA.attach("shm://{}".format(self.data_list[data_idx])).copy()
        data_idx = idx % len(self.data_list)
        data_path = self.data_list[data_idx]
        data = torch.load(data_path)

        coord, feat = data[0], data[1]
        if self.split != "test":
            label = data[2]

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
        # return len(self.data_idx) * self.loop
        return len(self.data_list) * self.loop


class Scannetv2_Point(Scannetv2):
    def __init__(
        self,
        split="train",
        data_root="trainval",
        voxel_size=0.04,
        voxel_max=None,
        transform=None,
        shuffle_index=False,
        loop=1,
        vote_num=1,
    ):
        super().__init__(
            split, data_root, voxel_size, voxel_max, transform, shuffle_index, loop
        )
        self.data_list = np.repeat(self.data_list, vote_num)
        print("Total repeated {} samples in {} set.".format(len(self.data_list), split))

    def __getitem__(self, idx):
        # data_idx = self.data_idx[idx % len(self.data_idx)]

        # data = SA.attach("shm://{}".format(self.data_list[data_idx])).copy()
        data_idx = idx % len(self.data_list)
        data_path = self.data_list[data_idx]
        data = torch.load(data_path)

        coord, feat = data[0], data[1]
        if self.split != "test":
            label = data[2]
        else:
            label = np.zeros(coord.shape[0], dtype=np.int32)

        coord, feat, label, label_pts, inverse_map = data_prepare_scannet_point(
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


class Scannetv2_Normal(Dataset):
    def __init__(
        self,
        split="train",
        data_root="trainval",
        voxel_size=0.04,
        voxel_max=None,
        transform=None,
        shuffle_index=False,
        loop=1,
        vote_num=1,
    ):
        super().__init__()

        self.split = split
        self.data_root = data_root
        self.voxel_size = voxel_size
        self.voxel_max = voxel_max
        self.transform = transform
        self.shuffle_index = shuffle_index
        self.loop = loop

        if split == "train" or split == "val":
            self.data_list = glob.glob(
                os.path.join(data_root, split, "*_vh_clean_2.ply")
            )
        elif split == "trainval":
            self.data_list = glob.glob(
                os.path.join(data_root, "train", "*_vh_clean_2.ply")
            ) + glob.glob(os.path.join(data_root, "val", "*_vh_clean_2.ply"))
        elif split == "trainval_v2":
            train_data_list = glob.glob(
                os.path.join(data_root, "train", "*_vh_clean_2.ply")
            )
            val_data_list = glob.glob(
                os.path.join(data_root, "val", "*_vh_clean_2.ply")
            )
            train_data_list.sort()
            val_data_list.sort()
            self.data_list = train_data_list + val_data_list[:-100]
            print("not include in train: ", val_data_list[-100:-90])
        elif split == "val_v2":
            val_data_list = glob.glob(
                os.path.join(data_root, "val", "*_vh_clean_2.ply")
            )
            val_data_list.sort()
            self.data_list = val_data_list[-100:]

        elif split == "val_v3":
            val_data_list = glob.glob(
                os.path.join(data_root, "val", "*_vh_clean_2.ply")
            )
            val_data_list.sort()
            print("val data: ", val_data_list[:-100])
            self.data_list = val_data_list[:-100]

        elif split == "test":
            self.data_list = glob.glob(
                os.path.join(data_root, split, "*_vh_clean_2.ply")
            )
            print(self.data_list)
        else:
            raise ValueError("no such split: {}".format(split))

        print("voxel_size: ", voxel_size)
        print("Totally {} samples in {} set.".format(len(self.data_list), split))

        self.data_list = np.repeat(self.data_list, vote_num)
        print("Total repeated {} samples in {} set.".format(len(self.data_list), split))

    def __getitem__(self, idx):
        # data_idx = self.data_idx[idx % len(self.data_idx)]

        # data = SA.attach("shm://{}".format(self.data_list[data_idx])).copy()
        data_idx = idx % len(self.data_list)
        data_path = self.data_list[data_idx]

        # data = read_plyfile(data_path)
        # v_color = data[:, 3:6] / 127.5 - 1
        pth_path = data_path[: -len("_vh_clean_2.ply")] + "_inst_nostuff.pth"
        data = torch.load(pth_path)
        v_color = data[1]
        if self.split != "test":
            v_label = data[2]
            # data_path_label = data_path[:-len(".ply")]+".labels.ply"
            # v_label = read_plyfile(data_path_label)[:, 7]
            # v_label = remapper[v_label.astype(np.int32)]
        else:
            v_label = np.zeros(v_color.shape[0])

        m = trimesh.load_mesh(data_path, process=False)
        v = m.vertices
        assert v.shape[0] == v_color.shape[0]
        n = m.vertex_normals

        coord = np.array(v, dtype=np.float32)
        coord = coord - coord.mean(axis=0)
        color_normal = np.concatenate([v_color, n], axis=1)
        feat = np.array(color_normal, dtype=np.float32)
        label = np.array(v_label, dtype=np.int32)
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
        # return len(self.data_idx) * self.loop
        return len(self.data_list) * self.loop


class Scannetv2_Normal_Point(Scannetv2_Normal):
    def __getitem__(self, idx):
        # data_idx = self.data_idx[idx % len(self.data_idx)]

        # data = SA.attach("shm://{}".format(self.data_list[data_idx])).copy()
        data_idx = idx % len(self.data_list)
        data_path = self.data_list[data_idx]

        # data = read_plyfile(data_path)
        # v_color = data[:, 3:6] / 127.5 - 1
        pth_path = data_path[: -len("_vh_clean_2.ply")] + "_inst_nostuff.pth"
        data = torch.load(pth_path)
        v_color = data[1]
        if self.split != "test":
            v_label = data[2]
            # data_path_label = data_path[:-len(".ply")]+".labels.ply"
            # v_label = read_plyfile(data_path_label)[:, 7]
            # v_label = remapper[v_label.astype(np.int32)]
        else:
            v_label = np.zeros(v_color.shape[0])

        m = trimesh.load_mesh(data_path, process=False)
        v = m.vertices
        assert v.shape[0] == v_color.shape[0]
        n = m.vertex_normals

        coord = np.array(v, dtype=np.float32)
        coord = coord - coord.mean(axis=0)
        color_normal = np.concatenate([v_color, n], axis=1)
        feat = np.array(color_normal, dtype=np.float32)
        label = np.array(v_label, dtype=np.int32)
        coord, feat, label, label_pts, inverse_map = data_prepare_scannet_point(
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
