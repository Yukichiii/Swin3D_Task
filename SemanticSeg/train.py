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
from datasets.s3dis import S3DIS, S3DIS_Point
from datasets.scannet_v2 import Scannetv2, Scannetv2_Point, Scannetv2_Normal, Scannetv2_Normal_Point
from util.common_util import AverageMeter, intersectionAndUnionGPU, find_free_port, poly_learning_rate, smooth_loss
from util.data_util import collate_fn, collate_fn_limit, collate_fn_pts
from util import transform
from util.logger import get_logger

from functools import partial
from util.lr import MultiStepWithWarmup, PolyLR, PolyLRwithWarmup
import MinkowskiEngine as ME
from torch.profiler import profile, record_function, ProfilerActivity
import importlib

def get_parser():
    parser = argparse.ArgumentParser(description='PyTorch Point Cloud Semantic Segmentation')
    parser.add_argument('--config', type=str, default='config/scannetv2/scannetv2_sparseswin.yaml', help='config file')
    parser.add_argument('opts', help='see config/scannetv2/scannetv2_sparseswin.yaml for all options', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    assert args.config is not None
    cfg = config.load_cfg_from_cfg_file(args.config)
    if args.opts is not None:
        cfg = config.merge_cfg_from_list(cfg, args.opts)
    if cfg.manual_seed is not None:
        cfg.save_path += "_"+str(cfg.manual_seed)
    return cfg


def worker_init_fn(worker_id):
    random.seed(args.manual_seed + worker_id)


def main_process():
    return not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % args.ngpus_per_node == 0)


def main():
    args = get_parser()
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in args.train_gpu)
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

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
        mp.spawn(main_worker, nprocs=args.ngpus_per_node, args=(args.ngpus_per_node, args))
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
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank)

    try:
        arch_name = args.arch
        model_module = importlib.import_module("model.%s"%arch_name)
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
            fp16_mode=args.get("fp16_mode", 0))
    
    # set loss func 
    criterion = nn.CrossEntropyLoss(ignore_index=args.ignore_label).cuda()
    
    # set optimizer
    if args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.base_lr, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optimizer == 'AdamW':
        transformer_lr_scale = args.get("transformer_lr_scale", 0.1)
        param_dicts = [
            {"params": [p for n, p in model.named_parameters() if "blocks" not in n and p.requires_grad]},
            {
                "params": [p for n, p in model.named_parameters() if "blocks" in n and p.requires_grad],
                "lr": args.base_lr * transformer_lr_scale,
            },
        ]
        optimizer = torch.optim.AdamW(param_dicts, lr=args.base_lr, weight_decay=args.weight_decay)

    if main_process():
        global logger, writer
        logger = get_logger(args.save_path)
        writer = SummaryWriter(args.save_path)
        logger.info(args)
        logger.info("=> creating model ...")
        logger.info("Classes: {}".format(args.classes))
        logger.info(model)
        logger.info('#Model parameters: {}'.format(sum([x.nelement() for x in model.parameters()])))

    if args.weight:
        if os.path.isfile(args.weight):
            if main_process():
                logger.info("=> loading weight '{}'".format(args.weight))
            model.backbone.load_pretrained_model(args.weight, skip_first_conv=args.get("skip_first_conv", True), verbose=main_process())
            if main_process():
                logger.info("=> loaded weight '{}'".format(args.weight))
        else:
            logger.info("=> no weight found at '{}'".format(args.weight))


    if args.distributed:
        torch.cuda.set_device(gpu)
        args.batch_size = int(args.batch_size / ngpus_per_node)
        args.batch_size_val = int(args.batch_size_val / ngpus_per_node)
        args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
        if args.sync_bn:
            if main_process():
                logger.info("use SyncBN")
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).cuda()
            model = ME.MinkowskiSyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu], find_unused_parameters=True)
    else:
        model = model.cuda()

    if args.resume:
        if args.resume == 'latest':
            args.resume = args.save_path + '/model/model_last.pth'
        if os.path.isfile(args.resume):
            if main_process():
                logger.info("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location=lambda storage, loc: storage.cuda())
            args.start_epoch = checkpoint['epoch']
            weights = checkpoint['state_dict']
            model.load_state_dict(weights, strict=True)
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler_state_dict = checkpoint['scheduler']
            best_iou = checkpoint['best_iou']
            if main_process():
                logger.info("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            if main_process():
                logger.info("=> no checkpoint found at '{}'".format(args.resume))

    if args.data_name == 's3dis':
        train_transform = None
        if args.aug:
            jitter_sigma = args.get('jitter_sigma', 0.01)
            jitter_clip = args.get('jitter_clip', 0.05)
            if main_process():
                logger.info("augmentation all")
                logger.info("jitter_sigma: {}, jitter_clip: {}".format(jitter_sigma, jitter_clip))
            train_transform = transform.Compose([
                transform.RandomRotate(along_z=args.get('rotate_along_z', True)),
                transform.RandomScale(scale_low=args.get('scale_low', 0.8), scale_high=args.get('scale_high', 1.2)),
                transform.RandomJitter(sigma=jitter_sigma, clip=jitter_clip),
                transform.RandomDropColor(color_augment=args.get('color_augment', 0.0)),
            ])
        train_data = S3DIS(split='train', data_root=args.data_root, test_area=args.test_area, voxel_size=args.voxel_size, voxel_max=args.voxel_max, transform=train_transform, shuffle_index=True, loop=args.loop)
    elif 'scannetv2' in args.data_name:
        train_transform = None
        if args.aug:
            if main_process():
                logger.info("use Augmentation")
            train_transform = transform.Compose([
                transform.RandomRotate(along_z=args.get('rotate_along_z', True)),
                transform.RandomScale(scale_low=args.get('scale_low', 0.8), scale_high=args.get('scale_high', 1.2)),
                transform.RandomDropColor(p=1-args.get('drop_color', 0.2), color_augment=args.get('color_augment', 0.0))
            ])
            
        train_split = args.get("train_split", "train")
        if main_process():
            logger.info("scannet. train_split: {}".format(train_split))

        if args.data_name == 'scannetv2':
            train_data = Scannetv2(split=train_split, data_root=args.data_root, voxel_size=args.voxel_size, voxel_max=args.voxel_max, transform=train_transform, shuffle_index=True, loop=args.loop)
        elif args.data_name == 'scannetv2_normal':
            train_data = Scannetv2_Normal(split=train_split, data_root=args.data_root, voxel_size=args.voxel_size, voxel_max=args.voxel_max, transform=train_transform, shuffle_index=True, loop=args.loop)
    else:
        raise ValueError("The dataset {} is not supported.".format(args.data_name))

    if main_process():
            logger.info("train_data samples: '{}'".format(len(train_data)))
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
    else:
        train_sampler = None
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=(train_sampler is None), num_workers=args.workers, \
        pin_memory=True, sampler=train_sampler, drop_last=True, collate_fn=partial(collate_fn_limit, max_batch_points=args.max_batch_points, logger=logger if main_process() else None))

    val_transform = None
    if args.data_name == 's3dis':
        val_data = S3DIS_Point(split='val', data_root=args.data_root, test_area=args.test_area, voxel_size=args.voxel_size, voxel_max=800000, transform=val_transform)
    elif 'scannetv2' in args.data_name:
        val_split = args.get("val_split", "val")
        if main_process():
            logger.info("scannet. val_split: {}".format(val_split))
        if args.data_name == 'scannetv2':
            val_data = Scannetv2_Point(split=val_split, data_root=args.data_root, voxel_size=args.voxel_size, voxel_max=800000, transform=val_transform)
        elif args.data_name == 'scannetv2_normal':
            val_data = Scannetv2_Normal_Point(split=val_split, data_root=args.data_root, voxel_size=args.voxel_size, voxel_max=800000, transform=val_transform)
    else:
        raise ValueError("The dataset {} is not supported.".format(args.data_name))

    if args.distributed:
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_data)
    else:
        val_sampler = None
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size_val, shuffle=False, num_workers=args.workers, \
            pin_memory=True, sampler=val_sampler, collate_fn=collate_fn_pts)
    
    # set scheduler
    if args.scheduler == "MultiStepWithWarmup":
        assert args.scheduler_update == 'step'
        if main_process():
            logger.info("scheduler: MultiStepWithWarmup. scheduler_update: {}".format(args.scheduler_update))
        iter_per_epoch = len(train_loader)
        milestones = [int(args.epochs*0.6) * iter_per_epoch, int(args.epochs*0.8) * iter_per_epoch]
        scheduler = MultiStepWithWarmup(optimizer, milestones=milestones, gamma=0.1, warmup=args.warmup, \
            warmup_iters=args.warmup_iters, warmup_ratio=args.warmup_ratio)
    elif args.scheduler == 'MultiStep':
        assert args.scheduler_update == 'epoch'
        milestones = [int(x) for x in args.milestones.split(",")] if hasattr(args, "milestones") else [int(args.epochs*0.6), int(args.epochs*0.8)]
        gamma = args.gamma if hasattr(args, 'gamma') else 0.1
        if main_process():
            logger.info("scheduler: MultiStep. scheduler_update: {}. milestones: {}, gamma: {}".format(args.scheduler_update, milestones, gamma))
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
    elif args.scheduler == 'Poly':
        if main_process():
            logger.info("scheduler: Poly. scheduler_update: {}".format(args.scheduler_update))
        if args.scheduler_update == 'epoch':
            scheduler = PolyLR(optimizer, max_iter=args.epochs, power=args.power)
        elif args.scheduler_update == 'step':
            iter_per_epoch = len(train_loader)
            scheduler = PolyLR(optimizer, max_iter=args.epochs*iter_per_epoch, power=args.power)
        else:
            raise ValueError("No such scheduler update {}".format(args.scheduler_update))
    elif args.scheduler == 'CosineWithWarmup':
        assert args.scheduler_update == 'step'
        if main_process():
            logger.info("scheduler: CosineWithWarmup. scheduler_update: {}".format(args.scheduler_update))    
        iter_per_epoch = len(train_loader)
        scheduler = CosineLRScheduler(optimizer,
                                  t_initial=args.epochs*iter_per_epoch,
                                  t_mul=1,
                                  lr_min=1e-6,
                                  decay_rate=0.1,
                                  warmup_lr_init=1e-6,
                                  warmup_t=args.warmup_iters,
                                  cycle_limit=1,
                                  t_in_epochs=True
        )
    else:
        raise ValueError("No such scheduler {}".format(args.scheduler))

    if args.resume and os.path.isfile(args.resume):
        scheduler.load_state_dict(scheduler_state_dict)
        print("resume scheduler")

    ###################
    # start training #
    ###################

    if args.use_amp:
        scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None
    
    
    #loss_val, mIoU_val, mAcc_val, allAcc_val = validate(val_loader, model, criterion)
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        if main_process():
            if isinstance(scheduler, CosineLRScheduler):
                logger.info("lr: {}".format(optimizer.param_groups[0]['lr']))
            else:
                logger.info("lr: {}".format(scheduler.get_last_lr()))
            
        loss_train, mIoU_train, mAcc_train, allAcc_train = train(train_loader, model, criterion, optimizer, epoch, scaler, scheduler)
        if args.scheduler_update == 'epoch':
            if isinstance(scheduler, CosineLRScheduler):
                scheduler.step(epoch)
            else:
                scheduler.step()
        epoch_log = epoch + 1
        
        if main_process():
            writer.add_scalar('loss_train', loss_train, epoch_log)
            writer.add_scalar('mIoU_train', mIoU_train, epoch_log)
            writer.add_scalar('mAcc_train', mAcc_train, epoch_log)
            writer.add_scalar('allAcc_train', allAcc_train, epoch_log)

        is_best = False
        if args.evaluate and (epoch_log % args.eval_freq == 0):
            loss_val, mIoU_val, mAcc_val, allAcc_val = validate(val_loader, model, criterion)

            if main_process():
                writer.add_scalar('loss_val', loss_val, epoch_log)
                writer.add_scalar('mIoU_val', mIoU_val, epoch_log)
                writer.add_scalar('mAcc_val', mAcc_val, epoch_log)
                writer.add_scalar('allAcc_val', allAcc_val, epoch_log)
                is_best = mIoU_val > best_iou
                best_iou = max(best_iou, mIoU_val)

        if (epoch_log % args.save_freq == 0) and main_process():
            if not os.path.exists(args.save_path + "/model/"):
                os.makedirs(args.save_path + "/model/")
            filename = args.save_path + '/model/model_last.pth'
            logger.info('Saving checkpoint to: ' + filename)
            torch.save({'epoch': epoch_log, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(), 'best_iou': best_iou, 'is_best': is_best}, filename)
            if is_best:
                shutil.copyfile(filename, args.save_path + '/model/model_best.pth')

    if main_process():
        writer.close()
        logger.info('==>Training done!\nBest Iou: %.3f' % (best_iou))


def train(train_loader, model, criterion, optimizer, epoch, scaler, scheduler):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()
    model.train()
    end = time.time()
    max_iter = args.epochs * len(train_loader)
    for i, (coord, feat, target, offset) in enumerate(train_loader):  # (n, 3), (n, c), (n), (b)
        data_time.update(time.time() - end)


        # some XYZ-RGB pretrained models are y-upright
        if args.get("yz_shift", False):
            coord = coord[:, [0, 2, 1]]
            if 'normal' in args.data_name:
                feat = feat[:, [0,1,2,3,5,4]]

        offset_ = offset.clone()
        offset_[1:] = offset_[1:] - offset_[:-1]
        batch = torch.cat([torch.tensor([ii]*o) for ii,o in enumerate(offset_)], 0).long()

        coord, feat, target, offset = coord.cuda(non_blocking=True), feat.cuda(non_blocking=True), target.cuda(non_blocking=True), offset.cuda(non_blocking=True)
        batch = batch.cuda(non_blocking=True)
        assert batch.shape[0] == feat.shape[0]
        
        if args.concat_xyz:
            feat = torch.cat([feat, coord], 1)

        use_amp = args.use_amp
        with torch.cuda.amp.autocast(enabled=use_amp):
            output = model(feat, coord, batch)
            assert output.shape[1] == args.classes
            if target.shape[-1] == 1:
                target = target[:, 0]  # for cls
            loss = criterion(output, target)
        
        optimizer.zero_grad()
        
        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        if args.scheduler_update == 'step':
            if isinstance(scheduler, CosineLRScheduler):
                scheduler.step(epoch*len(train_loader)+i)
            else:
                scheduler.step()

        output = output.max(1)[1]
        n = coord.size(0)
        if args.multiprocessing_distributed:
            loss *= n
            count = target.new_tensor([n], dtype=torch.long)
            dist.all_reduce(loss), dist.all_reduce(count)
            n = count.item()
            loss /= n
        intersection, union, target = intersectionAndUnionGPU(output, target, args.classes, args.ignore_label)
        if args.multiprocessing_distributed:
            dist.all_reduce(intersection), dist.all_reduce(union), dist.all_reduce(target)
        intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()
        intersection_meter.update(intersection), union_meter.update(union), target_meter.update(target)

        accuracy = sum(intersection_meter.val) / (sum(target_meter.val) + 1e-10)
        loss_meter.update(loss.item(), n)
        batch_time.update(time.time() - end)
        end = time.time()

        # calculate remain time
        current_iter = epoch * len(train_loader) + i + 1
        remain_iter = max_iter - current_iter
        remain_time = remain_iter * batch_time.avg
        t_m, t_s = divmod(remain_time, 60)
        t_h, t_m = divmod(t_m, 60)
        remain_time = '{:02d}:{:02d}:{:02d}'.format(int(t_h), int(t_m), int(t_s))

        if (i + 1) % args.print_freq == 0 and main_process():
            if isinstance(scheduler, CosineLRScheduler):
                lr = [param_group['lr'] for param_group in optimizer.param_groups]
            else:
                lr = scheduler.get_last_lr()
            if isinstance(lr, list):
                lr = [round(x, 8) for x in lr]
            elif isinstance(lr, float):
                lr = round(lr, 8)
            logger.info('Epoch: [{}/{}][{}/{}] '
                        'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                        'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                        'Remain {remain_time} '
                        'Loss {loss_meter.val:.4f} '
                        'Lr: {lr} '
                        'Accuracy {accuracy:.4f}.'.format(epoch+1, args.epochs, i + 1, len(train_loader),
                                                          batch_time=batch_time, data_time=data_time,
                                                          remain_time=remain_time,
                                                          loss_meter=loss_meter,
                                                          lr=lr,
                                                          accuracy=accuracy))
        if main_process():
            writer.add_scalar('loss_train_batch', loss_meter.val, current_iter)
            writer.add_scalar('mIoU_train_batch', np.mean(intersection / (union + 1e-10)), current_iter)
            writer.add_scalar('mAcc_train_batch', np.mean(intersection / (target + 1e-10)), current_iter)
            writer.add_scalar('allAcc_train_batch', accuracy, current_iter)

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)
    mAcc = np.mean(accuracy_class)
    allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)
    if main_process():
        logger.info('Train result at epoch [{}/{}]: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(epoch+1, args.epochs, mIoU, mAcc, allAcc))
    torch.cuda.empty_cache()
    return loss_meter.avg, mIoU, mAcc, allAcc


def validate(val_loader, model, criterion):
    if main_process():
        logger.info('>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>')
    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()

    intersection_meter_pts = AverageMeter()
    union_meter_pts = AverageMeter()
    target_meter_pts = AverageMeter()

    torch.cuda.empty_cache()

    model.eval()
    if args.distributed:
        classifier = model.module
    else:
        classifier = model

    classifier = classifier.backbone.classifier

    cls_find_nan = False
    for k,v in classifier.state_dict().items():
        if torch.isnan(v).any():
            print(k, "is nan!")
            cls_find_nan = True
    if cls_find_nan:
        print("set classifier to train mode!")
        classifier.train()
    else:
        print("keep classifier in val mode!")
    end = time.time()
    for i, (coord, feat, target, offset, target_pts, inverse_map) in enumerate(val_loader):
        data_time.update(time.time() - end)

        # some XYZ-RGB pretrained models are y-upright
        if args.get("yz_shift", False):
            coord = coord[:, [0, 2, 1]]
            if 'normal' in args.data_name:
                feat = feat[:, [0,1,2,3,5,4]]


        offset_ = offset.clone()
        offset_[1:] = offset_[1:] - offset_[:-1]
        batch = torch.cat([torch.tensor([ii]*o) for ii,o in enumerate(offset_)], 0).long()

        coord, feat, target, offset = coord.cuda(non_blocking=True), feat.cuda(non_blocking=True), target.cuda(non_blocking=True), offset.cuda(non_blocking=True)
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
            loss = criterion(output, target)
            output_pts = output[inverse_map]
        output = output.max(1)[1]
        n = coord.size(0)
        if args.multiprocessing_distributed:
            loss *= n
            count = target.new_tensor([n], dtype=torch.long)
            dist.all_reduce(loss), dist.all_reduce(count)
            n = count.item()
            loss /= n

        intersection, union, target = intersectionAndUnionGPU(output, target, args.classes, args.ignore_label)
        if args.multiprocessing_distributed:
            dist.all_reduce(intersection), dist.all_reduce(union), dist.all_reduce(target)
        intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()
        intersection_meter.update(intersection), union_meter.update(union), target_meter.update(target)

        accuracy = sum(intersection_meter.val) / (sum(target_meter.val) + 1e-10)
        loss_meter.update(loss.item(), n)
        batch_time.update(time.time() - end)
        end = time.time()
        if (i + 1) % args.print_freq == 0 and main_process():
            logger.info('Test: [{}/{}] '
                        'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                        'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                        'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f}) '
                        'Accuracy {accuracy:.4f}.'.format(i + 1, len(val_loader),
                                                          data_time=data_time,
                                                          batch_time=batch_time,
                                                          loss_meter=loss_meter,
                                                          accuracy=accuracy))

        output = output_pts.max(1)[1]
        target = target_pts
        intersection, union, target = intersectionAndUnionGPU(output, target, args.classes, args.ignore_label)
        if args.multiprocessing_distributed:
            dist.all_reduce(intersection), dist.all_reduce(union), dist.all_reduce(target)
        intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()
        intersection_meter_pts.update(intersection), union_meter_pts.update(union), target_meter_pts.update(target)

        accuracy = sum(intersection_meter_pts.val) / (sum(target_meter_pts.val) + 1e-10)
        end = time.time()
        if (i + 1) % args.print_freq == 0 and main_process():
            logger.info('Test Points: [{}/{}] '
                        'Accuracy {accuracy:.4f}.'.format(i + 1, len(val_loader),
                                                          accuracy=accuracy))

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)
    mAcc = np.mean(accuracy_class)
    allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)
    if main_process():
        logger.info('Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(mIoU, mAcc, allAcc))
        for i in range(args.classes):
            logger.info('Class_{} Result: iou/accuracy {:.4f}/{:.4f}.'.format(i, iou_class[i], accuracy_class[i]))
        logger.info('<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<')


    iou_class = intersection_meter_pts.sum / (union_meter_pts.sum + 1e-10)
    accuracy_class = intersection_meter_pts.sum / (target_meter_pts.sum + 1e-10)
    mIoU = np.mean(iou_class)
    mAcc = np.mean(accuracy_class)
    allAcc = sum(intersection_meter_pts.sum) / (sum(target_meter_pts.sum) + 1e-10)
    if main_process():
        logger.info('Val result Point: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(mIoU, mAcc, allAcc))
        for i in range(args.classes):
            logger.info('Class_{} Result: iou/accuracy {:.4f}/{:.4f}.'.format(i, iou_class[i], accuracy_class[i]))
        logger.info('<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<')

    torch.cuda.empty_cache()
    return loss_meter.avg, mIoU, mAcc, allAcc


if __name__ == '__main__':
    import gc
    gc.collect()
    main()
