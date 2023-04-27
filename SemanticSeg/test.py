import os
import time
import random
import numpy as np
import logging
import argparse
import shutil

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.optim.lr_scheduler as lr_scheduler
from timm.scheduler import CosineLRScheduler
from torch.utils.tensorboard import SummaryWriter

from util import config
from datasets.s3dis import S3DIS_Point
from datasets.scannet_v2 import Scannetv2_Point, Scannetv2_Normal_Point
from util.common_util import (
    AverageMeter,
    intersectionAndUnionGPU,
    find_free_port,
    poly_learning_rate,
    smooth_loss,
)
from util.data_util import collate_fn_pts
from util import transform
from util.logger import get_logger

from functools import partial
from util.lr import MultiStepWithWarmup, PolyLR, PolyLRwithWarmup
import MinkowskiEngine as ME
from torch.profiler import profile, record_function, ProfilerActivity
import importlib


def get_parser():
    parser = argparse.ArgumentParser(
        description="PyTorch Point Cloud Semantic Segmentation"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/scannetv2/scannetv2_sparseswin.yaml",
        help="config file",
    )
    parser.add_argument("--vote_num", type=int, default=1, help="rotation vote num")
    parser.add_argument(
        "--save_output", action="store_true", default=False, help="whether store output"
    )
    parser.add_argument(
        "opts",
        help="see config/scannetv2/scannetv2_sparseswin.yaml for all options",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()
    assert args.config is not None
    cfg = config.load_cfg_from_cfg_file(args.config)
    if args.opts is not None:
        cfg = config.merge_cfg_from_list(cfg, args.opts)
    cfg.vote_num = args.vote_num
    cfg.save_output = args.save_output
    cfg.save_path = cfg.weight.split(".")[0]
    return cfg


def worker_init_fn(worker_id):
    random.seed(args.manual_seed + worker_id)


def main_process():
    return not args.multiprocessing_distributed or (
        args.multiprocessing_distributed and args.rank % args.ngpus_per_node == 0
    )


def main():
    args = get_parser()
    args.train_gpu = args.train_gpu[:1]
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(x) for x in args.train_gpu)
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    # import torch.backends.mkldnn
    # ackends.mkldnn.enabled = False
    # os.environ["LRU_CACHE_CAPACITY"] = "1"
    # cudnn.deterministic = True
    if args.manual_seed is not None:
        random.seed(args.manual_seed)
        np.random.seed(args.manual_seed)
        torch.manual_seed(args.manual_seed)
        torch.cuda.manual_seed(args.manual_seed)
        torch.cuda.manual_seed_all(args.manual_seed)
        cudnn.benchmark = False
        cudnn.deterministic = True
    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])
    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    args.ngpus_per_node = len(args.train_gpu)
    if len(args.train_gpu) == 1:
        args.sync_bn = False
        args.distributed = False
        args.multiprocessing_distributed = False

    if args.multiprocessing_distributed:
        port = find_free_port()
        args.dist_url = f"tcp://127.0.0.1:{port}"
        args.world_size = args.ngpus_per_node * args.world_size
        mp.spawn(
            main_worker, nprocs=args.ngpus_per_node, args=(args.ngpus_per_node, args)
        )
    else:
        main_worker(args.train_gpu, args.ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, argss):
    global args, best_iou
    args, best_iou = argss, 0
    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(
            backend=args.dist_backend,
            init_method=args.dist_url,
            world_size=args.world_size,
            rank=args.rank,
        )

    # get model
    try:
        arch_name = args.arch
        model_module = importlib.import_module("model.%s" % arch_name)
    except:
        print("[ERROR]: model not found:", args.arch)
        raise NotImplementedError

    # get model
    model = model_module.Swin3D(
        depths=args.depths,
        channels=args.channels,
        num_heads=args.num_heads,
        window_sizes=args.window_size,
        up_k=args.up_k,
        quant_sizes=args.quant_size,
        drop_path_rate=args.drop_path_rate,
        num_classes=args.classes,
        num_layers=args.num_layers,
        stem_transformer=args.stem_transformer,
        upsample=args.upsample,
        down_stride=args.get("down_stride", 2),
        knn_down=args.get("knn_down", True),
        signal=args.get("signal", True),
        in_channels=args.get("fea_dim", 6),
        use_offset=args.get("use_offset", False),
        fp16_mode=args.get("fp16_mode", 1),
    )

    global logger, writer
    logger = get_logger(args.save_path + "/eval_pts")
    logger.info(args)
    logger.info("=> creating model ...")
    logger.info("Classes: {}".format(args.classes))
    logger.info(model)
    logger.info(
        "#Model parameters: {}".format(sum([x.nelement() for x in model.parameters()]))
    )

    model = model.cuda()

    assert (".pth" in args.weight) and os.path.isfile(args.weight)

    if args.weight:
        if os.path.isfile(args.weight):
            if main_process():
                logger.info("=> loading weight '{}'".format(args.weight))
            checkpoint = torch.load(args.weight)
            weights = checkpoint["state_dict"]
            if list(weights.keys())[0].startswith("module."):
                logger.info("=> Loading multigpu weights with module. prefix...")
                weights = {
                    k.partition("module.")[2]: weights[k] for k in weights.keys()
                }
            model.backbone.load_state_dict(weights)
            if main_process():
                logger.info(
                    "=> loaded weight '{}' (epoch {})".format(
                        args.weight, checkpoint["epoch"]
                    )
                )
        else:
            logger.info("=> no weight found at '{}'".format(args.weight))

    vote_num = args.vote_num
    if vote_num > 1:
        val_transform = transform.Compose(
            [
                transform.RandomRotate(along_z=args.get("rotate_along_z", True)),
            ]
        )
    else:
        val_transform = None

    val_split = args.get("val_split", "val")
    if args.data_name == "scannetv2":
        val_data = Scannetv2_Point(
            split=val_split,
            data_root=args.data_root,
            voxel_size=args.voxel_size,
            voxel_max=800000,
            transform=val_transform,
            vote_num=vote_num,
        )
    elif args.data_name == "scannetv2_normal":
        val_data = Scannetv2_Normal_Point(
            split=val_split,
            data_root=args.data_root,
            voxel_size=args.voxel_size,
            voxel_max=800000,
            transform=val_transform,
            vote_num=vote_num,
        )
    elif args.data_name == "s3dis":
        val_data = S3DIS_Point(
            split=val_split,
            data_root=args.data_root,
            test_area=args.test_area,
            voxel_size=args.voxel_size,
            voxel_max=800000,
            transform=val_transform,
            vote_num=vote_num,
        )

    val_loader = torch.utils.data.DataLoader(
        val_data,
        batch_size=1,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        sampler=None,
        collate_fn=collate_fn_pts,
    )

    validate(val_loader, model, vote_num)


import torch.nn.functional as F


def voted_output(model, data_iter, vote_num):
    output_pts_voted = 0

    for it in range(vote_num):
        coord, feat, target, offset, target_pts, inverse_map = data_iter.next()

        if args.get("yz_shift", False):
            coord = coord[:, [0, 2, 1]]
            if "normal" in args.data_name:
                feat = feat[:, [0, 1, 2, 3, 5, 4]]

        offset_ = offset.clone()
        offset_[1:] = offset_[1:] - offset_[:-1]
        batch = torch.cat(
            [torch.tensor([ii] * o) for ii, o in enumerate(offset_)], 0
        ).long()

        coord, feat, target, offset = (
            coord.cuda(non_blocking=True),
            feat.cuda(non_blocking=True),
            target.cuda(non_blocking=True),
            offset.cuda(non_blocking=True),
        )
        batch = batch.cuda(non_blocking=True)
        inverse_map = inverse_map.cuda(non_blocking=True)
        target_pts = target_pts.cuda(non_blocking=True)

        assert batch.shape[0] == feat.shape[0]

        if target.shape[-1] == 1:
            target = target[:, 0]  # for cls
            target_pts = target_pts[:, 0]  # for cls

        if args.concat_xyz:
            feat = torch.cat([feat, coord], 1)

        with torch.no_grad():
            output = model(feat, coord, batch)
            output_pts = output[inverse_map]
            output_pts = F.normalize(output_pts, p=1, dim=1)

        output_pts_voted += output_pts
    output_pts_voted /= vote_num
    return output_pts_voted, target_pts


def validate(val_loader, model, vote_num=12):
    if main_process():
        logger.info(">>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>")
    batch_time = AverageMeter()

    intersection_meter_pts = AverageMeter()
    union_meter_pts = AverageMeter()
    target_meter_pts = AverageMeter()

    torch.cuda.empty_cache()

    model.eval()
    classifier = model.backbone.classifier
    cls_find_nan = False
    for k, v in classifier.state_dict().items():
        if torch.isnan(v).any():
            print(k, "is nan!")
            cls_find_nan = True
    if cls_find_nan:
        print("set classifier to train mode!")
        classifier.train()
    else:
        print("keep classifier in val mode!")
    val_num = len(val_loader)
    print(val_num, vote_num)
    assert val_num % vote_num == 0
    val_num = int(val_num / vote_num)
    val_iter = iter(val_loader)

    if vote_num > 12:
        save_path = (
            args.weight.split(".")[0]
            + "/"
            + args.get("val_split", "val")
            + f"_vote{vote_num}"
        )
    else:
        save_path = args.weight.split(".")[0] + "/" + args.get("val_split", "val")
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    end = time.time()
    for i in range(val_num):
        output_pts, target_pts = voted_output(model, val_iter, vote_num)
        output = output_pts.max(1)[1]

        if args.save_output:
            output_np = output_pts.cpu().numpy()
            out_name = save_path + "/%03d.npy" % i
            np.save(out_name, output_np)
            print("save to ", out_name)
        if args.get("val_split", "val") == "test":
            continue

        target = target_pts
        intersection, union, target = intersectionAndUnionGPU(
            output, target, args.classes, args.ignore_label
        )
        if args.multiprocessing_distributed:
            dist.all_reduce(intersection), dist.all_reduce(union), dist.all_reduce(
                target
            )
        intersection, union, target = (
            intersection.cpu().numpy(),
            union.cpu().numpy(),
            target.cpu().numpy(),
        )
        intersection_meter_pts.update(intersection), union_meter_pts.update(
            union
        ), target_meter_pts.update(target)

        batch_time.update(time.time() - end)
        end = time.time()
        accuracy = sum(intersection_meter_pts.val) / (sum(target_meter_pts.val) + 1e-10)
        if (i + 1) % args.print_freq == 0 and main_process():
            logger.info(
                "Test Points: [{}/{}] "
                "Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) "
                "Accuracy {accuracy:.4f}.".format(
                    i + 1, val_num, batch_time=batch_time, accuracy=accuracy
                )
            )

    if args.get("val_split", "val") == "test":
        return

    iou_class = intersection_meter_pts.sum / (union_meter_pts.sum + 1e-10)
    accuracy_class = intersection_meter_pts.sum / (target_meter_pts.sum + 1e-10)
    mIoU = np.mean(iou_class)
    mAcc = np.mean(accuracy_class)
    allAcc = sum(intersection_meter_pts.sum) / (sum(target_meter_pts.sum) + 1e-10)
    if main_process():
        logger.info(
            "Val result Point: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.".format(
                mIoU, mAcc, allAcc
            )
        )
        for i in range(args.classes):
            logger.info(
                "Class_{:02d} Result: iou/accuracy {:.4f}/{:.4f}.".format(
                    i, iou_class[i], accuracy_class[i]
                )
            )
        logger.info("<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<")

    torch.cuda.empty_cache()
    return mIoU, mAcc, allAcc


if __name__ == "__main__":
    import gc

    gc.collect()
    main()
