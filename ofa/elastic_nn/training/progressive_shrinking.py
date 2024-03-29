# Once for All: Train One Network and Specialize it for Efficient Deployment
# Han Cai, Chuang Gan, Tianzhe Wang, Zhekai Zhang, Song Han
# International Conference on Learning Representations (ICLR), 2020.

import json
import torch.nn as nn
from tqdm import tqdm
import random
import os
import time

import torch
import torch.nn.functional as F
# import horovod.torch as hvd

from ofa.utils import accuracy, AverageMeter, download_url, psnr
from ofa.imagenet_codebase.utils import list_mean, cross_entropy_loss_with_soft_target, \
    subset_mean, int2list
from ofa.imagenet_codebase.data_providers.base_provider import MyRandomResizedCrop
# from ofa.imagenet_codebase.run_manager.distributed_run_manager import DistributedRunManager
from ofa.imagenet_codebase.run_manager.sr_run_manager import SRRunManager


def validate(run_manager, epoch=0, is_test=True, image_size_list=None,
             width_mult_list=None, ks_list=None, expand_ratio_list=None, depth_list=None, pixelshuffle_depth_list=None, additional_setting=None):
    dynamic_net = run_manager.net
    if isinstance(dynamic_net, nn.DataParallel):
        dynamic_net = dynamic_net.module

    dynamic_net.eval()

    if image_size_list is None:
        image_size_list = int2list(run_manager.run_config.data_provider.image_size, 1)
    if width_mult_list is None:
        width_mult_list = [i for i in range(len(dynamic_net.width_mult_list))]
    if ks_list is None:
        ks_list = dynamic_net.ks_list
    if expand_ratio_list is None:
        expand_ratio_list = dynamic_net.expand_ratio_list
    if depth_list is None:
        depth_list = dynamic_net.depth_list
    if pixelshuffle_depth_list is None:
        pixelshuffle_depth_list = dynamic_net.pixelshuffle_depth_list

    subnet_settings = []
    for pixel_d in pixelshuffle_depth_list:
        for w in width_mult_list:
                for d in depth_list:
                    for e in expand_ratio_list:
                        for k in ks_list:
                            # for img_size in image_size_list:
                            subnet_settings.append([{
                                # 'image_size': img_size,
                                'pixel_d': pixel_d,
                                'wid': w,
                                'd': d,
                                'e': e,
                                'ks': k,
                            }, 'PD%s-W%s-D%s-E%s-K%s' % (pixel_d, w, d, e, k)])
    if additional_setting is not None:
        subnet_settings += additional_setting

    # losses_of_subnets, top1_of_subnets, top5_of_subnets = [], [], []
    losses_of_subnets, psnr_of_subnets = [], []

    valid_log = ''
    for setting, name in subnet_settings:
        #################### Validation Architecture 정하는 부분인데, Single Architecture Overfitting 혹은 뭐 빠르게 테스트 해볼일 있으면 여기서 그냥 스킵하면됨
        # if name.find('PD1-W0-D2-E3-K7') == -1:
        #     continue

        run_manager.write_log('-' * 30 + ' Validate %s ' % name + '-' * 30, 'train', should_print=False)
        # run_manager.run_config.data_provider.assign_active_img_size(setting.pop('image_size'))

        #################### Random Sampling과 Structured Sampling중에 주석 바꿔가면서 고르면 됨
        # dynamic_net.sample_active_subnet()
        dynamic_net.set_active_subnet(**setting)
        
        
        run_manager.write_log(dynamic_net.module_str, 'train', should_print=False)

        #################### Oracle Training 시에는 Batch Mean/Variance 현재 데이터로 업데이트하면 망함.
        # run_manager.reset_running_statistics(dynamic_net)
        loss, psnr = run_manager.validate(epoch=epoch, is_test=is_test, run_str=name, net=dynamic_net)
        losses_of_subnets.append(loss)
        # top1_of_subnets.append(top1)
        # top5_of_subnets.append(top5)
        psnr_of_subnets.append(psnr)
        valid_log += '%s (%.3f), ' % (name, psnr)

    return list_mean(losses_of_subnets), list_mean(psnr_of_subnets), valid_log


def train_one_epoch(run_manager, args, epoch, warmup_epochs=0, warmup_lr=0):
    dynamic_net = run_manager.net
    if isinstance(dynamic_net, nn.DataParallel):
        dynamic_net = dynamic_net.module

    # switch to train mode
    dynamic_net.train()
    # run_manager.run_config.train_loader.sampler.set_epoch(epoch)
    # MyRandomResizedCrop.EPOCH = epoch
    #################### Code for freezing BN. Overfitting 할 때는 주석 해제하면됨.
    # for m in dynamic_net.modules():
    #     if isinstance(m, nn.BatchNorm2d):
    #         ########## Use running mean/var
    #         m.eval()
    #         ########## BN weight/bias freeze
    #         # m.weight.requires_grad = False
    #         # m.bias.requires_grad = False

    nBatch = len(run_manager.run_config.train_loader)

    data_time = AverageMeter()
    # losses = DistributedMetric('train_loss')
    # top1 = DistributedMetric('train_top1')
    # top5 = DistributedMetric('train_top5')
    losses = AverageMeter()
    psnr_averagemeter = AverageMeter()

    with tqdm(total=nBatch,
              desc='Train Epoch #{}'.format(epoch + 1)) as t:
        end = time.time()
        for i, mini_batch in enumerate(run_manager.run_config.train_loader):
            images = mini_batch['image']
            #################### 2x or 4x 고르는 부분.
            x2_down_images = mini_batch['2x_down_image']
            x4_down_images = mini_batch['4x_down_image']
            data_time.update(time.time() - end)
            if epoch < warmup_epochs:
                new_lr = run_manager.run_config.warmup_adjust_learning_rate(
                    run_manager.optimizer, warmup_epochs * nBatch, nBatch, epoch, i, warmup_lr,
                )
            else:
                new_lr = run_manager.run_config.adjust_learning_rate(
                    run_manager.optimizer, epoch - warmup_epochs, i, nBatch
                )

            images = images.cuda()
            #################### 2x or 4x 고르는 부분.
            x2_down_images = x2_down_images.cuda()
            x4_down_images = x4_down_images.cuda()
            target = images

            # soft target
            if args.kd_ratio > 0:
                args.teacher_model.train()
                with torch.no_grad():
                    soft_logits = args.teacher_model(images).detach()
                    soft_label = F.softmax(soft_logits, dim=1)

            # clear gradients
            run_manager.optimizer.zero_grad()

            loss_of_subnets, psnr_of_subnets = [], []
            # compute output
            subnet_str = ''
            for _ in range(args.dynamic_batch_size):

                # set random seed before sampling
                if args.independent_distributed_sampling:
                    subnet_seed = os.getpid() + time.time()
                else:
                    subnet_seed = int('%d%.3d%.3d' % (epoch * nBatch + i, _, 0))
                random.seed(subnet_seed)

                #################### Random Sampling과 Structured Sampling중에 주석 바꿔가면서 고르면 됨. Single Architecture Overfitting을 위해서 여기 수정해주면 가능.
                subnet_settings = dynamic_net.sample_active_subnet()
                # dynamic_net.set_active_subnet(ks=7, e=3, d=2, pixel_d=1)

                subnet_str += '%d: ' % _ + ','.join(['%s_%s' % (
                    key, '%.1f' % subset_mean(val, 0) if isinstance(val, list) else val
                ) for key, val in subnet_settings.items()]) + ' || '

                #################### 2x or 4x 고르는 부분.
                # output = run_manager.net(images)
                if subnet_settings['pixel_d'][0] == 1:
                    output = run_manager.net(x2_down_images)
                elif subnet_settings['pixel_d'][0] == 2:
                    output = run_manager.net(x4_down_images)

                if args.kd_ratio == 0:
                    loss = run_manager.train_criterion(output, images)
                    loss_type = 'mse'
                else:
                    if args.kd_type == 'ce':
                        kd_loss = cross_entropy_loss_with_soft_target(output, soft_label)
                    else:
                        kd_loss = F.mse_loss(output, soft_logits)
                    loss = args.kd_ratio * kd_loss + run_manager.train_criterion(output, labels)
                    loss = loss * (2 / (args.kd_ratio + 1))
                    loss_type = '%.1fkd-%s & ce' % (args.kd_ratio, args.kd_type)

                # measure accuracy and record loss
                # acc1, acc5 = accuracy(output, target, topk=(1, 5))
                psnr_current = psnr(rgb2y(tensor2img_np(output)), rgb2y(tensor2img_np(images)))
                loss_of_subnets.append(loss)
                # acc1_of_subnets.append(acc1[0])
                # acc5_of_subnets.append(acc5[0])
                psnr_of_subnets.append(psnr_current)

                loss.backward()
            run_manager.optimizer.step()

            losses.update(list_mean(loss_of_subnets), images.size(0))
            # top1.update(list_mean(acc1_of_subnets), images.size(0))
            # top5.update(list_mean(acc5_of_subnets), images.size(0))
            psnr_averagemeter.update(list_mean(psnr_of_subnets), images.size(0))

            t.set_postfix({
                'loss': losses.avg.item(),
                # 'top1': top1.avg.item(),
                # 'top5': top5.avg.item(),
                'psnr': psnr_averagemeter.avg,
                'R': images.size(2),
                'lr': new_lr,
                'loss_type': loss_type,
                'seed': str(subnet_seed),
                'str': subnet_str,
                'data_time': data_time.avg,
            })
            t.update(1)
            end = time.time()
    return losses.avg.item(), psnr_averagemeter.avg


def train(run_manager, args, validate_func=None):
    if validate_func is None:
        validate_func = validate

    for epoch in range(run_manager.start_epoch, run_manager.run_config.n_epochs + args.warmup_epochs):
        train_loss, train_top1 = train_one_epoch(
            run_manager, args, epoch, args.warmup_epochs, args.warmup_lr)

        if (epoch + 1) % args.validation_frequency == 0:
            # validate under train mode
            val_loss, val_acc, _val_log = validate_func(run_manager, epoch=epoch, is_test=True)
            # best_acc
            is_best = val_acc > run_manager.best_acc
            run_manager.best_acc = max(run_manager.best_acc, val_acc)
            # if run_manager.is_root:
            val_log = 'Valid [{0}/{1}] loss={2:.3f}, top-1={3:.3f} ({4:.3f})'. \
                format(epoch + 1 - args.warmup_epochs, run_manager.run_config.n_epochs, val_loss, val_acc,
                        run_manager.best_acc)
            val_log += ', Train top-1 {top1:.3f}, Train loss {loss:.3f}\t'.format(top1=train_top1, loss=train_loss)
            val_log += _val_log
            run_manager.write_log(val_log, 'valid', should_print=False)

            run_manager.save_model({
                'epoch': epoch,
                'best_acc': run_manager.best_acc,
                'optimizer': run_manager.optimizer.state_dict(),
                'state_dict': run_manager.net.state_dict(),
            }, is_best=is_best)


def load_models(run_manager, dynamic_net, model_path=None):
    # specify init path
    init = torch.load(model_path, map_location='cpu')['state_dict']
    if isinstance(dynamic_net, nn.DataParallel):
        dynamic_net = dynamic_net.module
    dynamic_net.load_weights_from_net(init)
    run_manager.write_log('Loaded init from %s' % model_path, 'valid')


def supporting_elastic_depth(train_func, run_manager, args, validate_func_dict):
    dynamic_net = run_manager.net
    if isinstance(dynamic_net, nn.DataParallel):
        dynamic_net = dynamic_net.module
        
    # load stage info
    stage_info_path = os.path.join(run_manager.path, 'depth.stage')
    try:
        stage_info = json.load(open(stage_info_path))
    except Exception:
        stage_info = {'stage': 0}
    
    # load pretrained models
    validate_func_dict['depth_list'] = sorted(dynamic_net.depth_list)

    if args.phase == 1:
        # model_path = download_url('https://hanlab.mit.edu/files/OnceForAll/ofa_checkpoints/ofa_D4_E6_K357',
        #                           model_dir='.torch/ofa_checkpoints/%d' % hvd.rank())
        model_path = './exp/sr/mbx4_bn_mse/teacher/checkpoint/model_best.pth.tar' #################### 필요에 맞춰서 바꿔줘야함
        load_models(run_manager, dynamic_net, model_path=model_path)
    else:
        # model_path = download_url('https://hanlab.mit.edu/files/OnceForAll/ofa_checkpoints/ofa_D34_E6_K357',
        #                           model_dir='.torch/ofa_checkpoints/%d' % hvd.rank())
        model_path = './exp/sr/mbx4_bn_mse/teacher/checkpoint/model_best.pth.tar' #################### 필요에 맞춰서 바꿔줘야함
        load_models(run_manager, dynamic_net, model_path=model_path)
    # validate after loading weights
    run_manager.write_log('%.3f\t%.3f\t%s' % validate(run_manager, **validate_func_dict), 'valid')
        
    depth_stage_list = dynamic_net.depth_list.copy()
    depth_stage_list.sort(reverse=True)
    n_stages = len(depth_stage_list) - 1
    start_stage = n_stages - 1

    for current_stage in range(start_stage, n_stages):
        run_manager.write_log(
            '-' * 30 + 'Supporting Elastic Depth: %s -> %s' %
            (depth_stage_list[:current_stage + 1], depth_stage_list[:current_stage + 2]) + '-' * 30, 'valid'
        )
        
        # add depth list constraints
        supported_depth = depth_stage_list[:current_stage + 2]
        if len(set(dynamic_net.ks_list)) == 1 and len(set(dynamic_net.expand_ratio_list)) == 1:
            validate_func_dict['depth_list'] = supported_depth
        else:
            validate_func_dict['depth_list'] = sorted({min(supported_depth), max(supported_depth)})
        dynamic_net.set_constraint(supported_depth, constraint_type='depth')
        
        # train
        train_func(
            run_manager, args,
            lambda _run_manager, epoch, is_test: validate(_run_manager, epoch, is_test, **validate_func_dict)
        )

        # next stage & reset
        stage_info['stage'] += 1
        run_manager.start_epoch = 0
        run_manager.best_acc = 0.0

        # save and validate
        run_manager.save_model(model_name='depth_stage%d.pth.tar' % stage_info['stage'])
        json.dump(stage_info, open(stage_info_path, 'w'), indent=4)
        validate_func_dict['depth_list'] = sorted(dynamic_net.depth_list)
        run_manager.write_log('%.3f\t%.3f\t%s' % validate(run_manager, **validate_func_dict), 'valid')


def supporting_elastic_expand(train_func, run_manager, args, validate_func_dict):
    dynamic_net = run_manager.net
    if isinstance(dynamic_net, nn.DataParallel):
        dynamic_net = dynamic_net.module

    # load stage info
    stage_info_path = os.path.join(run_manager.path, 'expand.stage')
    try:
        stage_info = json.load(open(stage_info_path))
    except Exception:
        stage_info = {'stage': 0}

    # load pretrained models
    validate_func_dict['expand_ratio_list'] = sorted(dynamic_net.expand_ratio_list)

    if args.phase == 1:
        # model_path = download_url('https://hanlab.mit.edu/files/OnceForAll/ofa_checkpoints/ofa_D234_E6_K357',
        #                           model_dir='.torch/ofa_checkpoints/%d' % hvd.rank())
        model_path = './exp/sr/mbx4_bn_mse/teacher/checkpoint/model_best.pth.tar' #################### 필요에 맞춰서 바꿔줘야함
        load_models(run_manager, dynamic_net, model_path=model_path)
    else:
        # model_path = download_url('https://hanlab.mit.edu/files/OnceForAll/ofa_checkpoints/ofa_D234_E46_K357',
        #                           model_dir='.torch/ofa_checkpoints/%d' % hvd.rank())
        model_path = './exp/sr/mbx4_bn_mse/teacher/checkpoint/model_best.pth.tar' #################### 필요에 맞춰서 바꿔줘야함
        load_models(run_manager, dynamic_net, model_path=model_path)
    dynamic_net.re_organize_middle_weights()
    run_manager.write_log('%.3f\t%.3f\t%s' % validate(run_manager, **validate_func_dict), 'valid')
    
    expand_stage_list = dynamic_net.expand_ratio_list.copy()
    expand_stage_list.sort(reverse=True)
    n_stages = len(expand_stage_list) - 1
    start_stage = n_stages - 1
    
    for current_stage in range(start_stage, n_stages):
        run_manager.write_log(
            '-' * 30 + 'Supporting Elastic Expand Ratio: %s -> %s' %
            (expand_stage_list[:current_stage + 1], expand_stage_list[:current_stage + 2]) + '-' * 30, 'valid'
        )
        
        # add expand list constraints
        supported_expand = expand_stage_list[:current_stage + 2]
        if len(set(dynamic_net.ks_list)) == 1 and len(set(dynamic_net.depth_list)) == 1:
            validate_func_dict['expand_ratio_list'] = supported_expand
        else:
            validate_func_dict['expand_ratio_list'] = sorted({min(supported_expand), max(supported_expand)})
        dynamic_net.set_constraint(supported_expand, constraint_type='expand_ratio')

        # train
        train_func(
            run_manager, args,
            lambda _run_manager, epoch, is_test: validate(_run_manager, epoch, is_test, **validate_func_dict)
        )

        # next stage & reset
        stage_info['stage'] += 1
        run_manager.start_epoch = 0
        run_manager.best_acc = 0.0
        dynamic_net.re_organize_middle_weights(expand_ratio_stage=stage_info['stage'])
        if isinstance(run_manager, DistributedRunManager):
            run_manager.broadcast()

        # save and validate
        run_manager.save_model(model_name='expand_stage%d.pth.tar' % stage_info['stage'])
        json.dump(stage_info, open(stage_info_path, 'w'), indent=4)
        validate_func_dict['expand_ratio_list'] = sorted(dynamic_net.expand_ratio_list)
        run_manager.write_log('%.3f\t%.3f\t%s' % validate(run_manager, **validate_func_dict), 'valid')


def supporting_elastic_pixelshuffle_depth(train_func, run_manager, args, validate_func_dict):
    dynamic_net = run_manager.net
    if isinstance(dynamic_net, nn.DataParallel):
        dynamic_net = dynamic_net.module
        
    # load stage info
    stage_info_path = os.path.join(run_manager.path, 'pixelshuffle_depth.stage')
    try:
        stage_info = json.load(open(stage_info_path))
    except Exception:
        stage_info = {'stage': 0}
    
    # load pretrained models
    validate_func_dict['pixelshuffle_depth_list'] = sorted(dynamic_net.pixelshuffle_depth_list)

    if args.phase == 1:
        # model_path = download_url('https://hanlab.mit.edu/files/OnceForAll/ofa_checkpoints/ofa_D4_E6_K357',
        #                           model_dir='.torch/ofa_checkpoints/%d' % hvd.rank())
        model_path = './complete/sr_bn_mse_4xLarge2pixelShuffle2readySetGo/checkpoint/model_best.pth.tar' #################### 필요에 맞춰서 바꿔줘야함
        load_models(run_manager, dynamic_net, model_path=model_path)
    else:
        # model_path = download_url('https://hanlab.mit.edu/files/OnceForAll/ofa_checkpoints/ofa_D34_E6_K357',
        #                           model_dir='.torch/ofa_checkpoints/%d' % hvd.rank())
        model_path = './complete/sr_bn_mse_4xLarge2pixelShuffle2readySetGo/checkpoint/model_best.pth.tar' #################### 필요에 맞춰서 바꿔줘야함
        load_models(run_manager, dynamic_net, model_path=model_path)
    # validate after loading weights
    run_manager.write_log('%.3f\t%.3f\t%s' % validate(run_manager, **validate_func_dict), 'valid')
        
    pixelshuffle_depth_stage_list = dynamic_net.pixelshuffle_depth_list.copy()
    pixelshuffle_depth_stage_list.sort(reverse=True)
    n_stages = len(pixelshuffle_depth_stage_list) - 1
    start_stage = n_stages - 1

    for current_stage in range(start_stage, n_stages):
        run_manager.write_log(
            '-' * 30 + 'Supporting Elastic Depth: %s -> %s' %
            (pixelshuffle_depth_stage_list[:current_stage + 1], pixelshuffle_depth_stage_list[:current_stage + 2]) + '-' * 30, 'valid'
        )
        
        # add depth list constraints
        supported_pixelshuffle_depth = pixelshuffle_depth_stage_list[:current_stage + 2]
        if len(set(dynamic_net.ks_list)) == 1 and len(set(dynamic_net.expand_ratio_list)) == 1:
            validate_func_dict['pixelshuffle_depth_list'] = supported_pixelshuffle_depth
        else:
            validate_func_dict['pixelshuffle_depth_list'] = sorted({min(supported_pixelshuffle_depth), max(supported_pixelshuffle_depth)})
        dynamic_net.set_constraint(supported_pixelshuffle_depth, constraint_type='pixelshuffle_depth')
        
        # train
        train_func(
            run_manager, args,
            lambda _run_manager, epoch, is_test: validate(_run_manager, epoch, is_test, **validate_func_dict)
        )

        # next stage & reset
        stage_info['stage'] += 1
        run_manager.start_epoch = 0
        run_manager.best_acc = 0.0

        # save and validate
        run_manager.save_model(model_name='pixelshuffle_depth_stage%d.pth.tar' % stage_info['stage'])
        json.dump(stage_info, open(stage_info_path, 'w'), indent=4)
        validate_func_dict['pixelshuffle_depth_list'] = sorted(dynamic_net.pixelshuffle_depth_list)
        run_manager.write_log('%.3f\t%.3f\t%s' % validate(run_manager, **validate_func_dict), 'valid')


import math
from PIL import Image
import numpy as np
import torch
from torchvision.utils import make_grid

"""
Converts a Tensor into an image Numpy array
Input should be either in 4D(B,(3/1),H,W), 3D(C,H,W), or 2D(H,W)
If input is in 4D, it is splited along the first dimension to provide grid view.
Otherwise, the tensor is assume to be single image.
Input type: float [-1, 1] (default)
Output type: np.uint8 [0,255] (default)
Output dim: 3D(H,W,C) (for 4D and 3D input) or 2D(H,W) (for 2D input)
"""
def tensor2img_np(tensor, out_type=np.uint8, min_max=(0,1)):
    tensor = tensor.float().cpu().clamp_(*min_max) # Clamp is for on hard_tanh
    tensor = (tensor - min_max[0]) / (min_max[1] - min_max[0])
    n_dim = tensor.dim()
    if n_dim == 4:
        n_img = len(tensor)
        img_np = make_grid(tensor, nrow=int(math.sqrt(n_img)), normalize=False).detach().numpy()
        img_np = np.transpose(img_np, (1, 2, 0))
    elif n_dim == 3:
        img_np = tensor.numpy()
        img_np = np.transpose(img_np, (1, 2, 0))
    elif n_dim == 2:
        img_np = tensor.numpy()
    else:
        raise TypeError('Only support 4D, 3D and 2D tensor. But receieved tensor with dimension = %d' % n_dim)
    if out_type == np.uint8:
        img_np = (img_np * 255.0).round() # This is important. Unlike matlab, numpy.unit8() WILL NOT round by default.
    return img_np.astype(out_type)

def rgb2gray(img):
    in_img_type = img.dtype
    img.astype(np.float64)
    img_gray = np.dot(img[...,:3], [0.299, 0.587, 0.114]).round()
    return img_gray.astype(in_img_type)

def rgb2y(img):
    assert(img.dtype == np.uint8)
    in_img_type = img.dtype
    img.astype(np.float64)
    img_y = ((np.dot(img[...,:3], [65.481, 128.553, 24.966])) / 255.0 + 16.0).round()
    return img_y.astype(in_img_type)