# Once for All: Train One Network and Specialize it for Efficient Deployment
# Han Cai, Chuang Gan, Tianzhe Wang, Zhekai Zhang, Song Han
# International Conference on Learning Representations (ICLR), 2020.

import argparse
import numpy as np
import os
import random

# import horovod.torch as hvd
import torch

from ofa.elastic_nn.modules.dynamic_op import DynamicSeparableConv2d
from ofa.elastic_nn.networks import OFAMobileNetS4
from ofa.imagenet_codebase.run_manager import Div2K_SetXXRunConfig
from ofa.imagenet_codebase.run_manager.sr_run_manager import SRRunManager
from ofa.imagenet_codebase.data_providers.base_provider import MyRandomResizedCrop  # SR 할때는 안씀, 그냥 여기서 Parameter 초기화하는데 빼기 귀찮아서 냅둠
from ofa.utils import download_url
from ofa.elastic_nn.training.progressive_shrinking import load_models

parser = argparse.ArgumentParser()
# parser.add_argument('--task', type=str, default='depth', choices=[
#     'kernel', 'depth', 'expand',
# ])
# parser.add_argument('--phase', type=int, default=1, choices=[1, 2])

args = parser.parse_args()
# if args.task == 'kernel':
#     args.path = 'exp/normal2kernel'
#     args.dynamic_batch_size = 1
#     args.n_epochs = 120
#     args.base_lr = 3e-2
#     args.warmup_epochs = 5
#     args.warmup_lr = -1
#     args.ks_list = '3,5,7'
#     args.expand_list = '6'
#     args.depth_list = '4'
# elif args.task == 'depth':
#     args.path = 'exp/kernel2kernel_depth/phase%d' % args.phase
#     args.dynamic_batch_size = 2
#     if args.phase == 1:
#         args.n_epochs = 25
#         args.base_lr = 2.5e-3
#         args.warmup_epochs = 0
#         args.warmup_lr = -1
#         args.ks_list = '3,5,7'
#         args.expand_list = '6'
#         args.depth_list = '3,4'
#     else:
#         args.n_epochs = 120
#         args.base_lr = 7.5e-3
#         args.warmup_epochs = 5
#         args.warmup_lr = -1
#         args.ks_list = '3,5,7'
#         args.expand_list = '6'
#         args.depth_list = '2,3,4'
# elif args.task == 'expand':
#     args.path = 'exp/kernel_depth2kernel_depth_width/phase%d' % args.phase
#     args.dynamic_batch_size = 4
#     if args.phase == 1:
#         args.n_epochs = 25
#         args.base_lr = 2.5e-3
#         args.warmup_epochs = 0
#         args.warmup_lr = -1
#         args.ks_list = '3,5,7'
#         args.expand_list = '4,6'
#         args.depth_list = '2,3,4'
#     else:
#         args.n_epochs = 120
#         args.base_lr = 7.5e-3
#         args.warmup_epochs = 5
#         args.warmup_lr = -1
#         args.ks_list = '3,5,7'
#         args.expand_list = '3,4,6'
#         args.depth_list = '2,3,4'
# else:
#     raise NotImplementedError
args.path = 'exp/sr_x2_k5_e3_d2_bn_mse_tune_low'
args.n_epochs = 10
args.base_lr = 0.0001  # Default (Worked Well): 0.001
args.warmup_epochs = 5
args.warmup_lr = -1
args.ks_list = '5'
args.expand_list = '3'
args.depth_list = '2'
args.pixelshuffle_depth_list = '1'
args.manual_seed = 0

args.lr_schedule_type = 'cosine'

args.base_batch_size = 4  # Default (Worked Well): 16
args.valid_size = None

args.opt_type = 'adam'
args.momentum = 0.9
args.no_nesterov = False
args.weight_decay = 3e-5
args.label_smoothing = 0.0
args.no_decay_keys = 'bn#bias'
args.fp16_allreduce = False

args.model_init = 'he_fout'
args.validation_frequency = 1
args.print_frequency = 10

args.n_worker = 8
args.resize_scale = 1.0
args.distort_color = None
args.image_size = '480'
args.continuous_size = True
args.not_sync_distributed_image_size = False

args.bn_momentum = 0.1
args.bn_eps = 1e-5
args.dropout = 0.1
args.base_stage_width = 'proxyless'

args.width_mult_list = '1.0'
args.dy_conv_scaling_mode = 1
args.independent_distributed_sampling = False

args.kd_ratio = 0.0
args.kd_type = None

args.num_gpus = 4


if __name__ == '__main__':
    os.makedirs(args.path, exist_ok=True)

    # Initialize Horovod
    # hvd.init()
    # Pin GPU to be used to process local rank (one GPU per process)
    # torch.cuda.set_device(hvd.local_rank())

    # args.teacher_path = download_url(
    #     'https://hanlab.mit.edu/files/OnceForAll/ofa_checkpoints/ofa_D4_E6_K7',
    #     model_dir='.torch/ofa_checkpoints/%d' % hvd.rank()
    # )

    num_gpus = args.num_gpus

    torch.manual_seed(args.manual_seed)
    torch.cuda.manual_seed_all(args.manual_seed)
    np.random.seed(args.manual_seed)
    random.seed(args.manual_seed)

    # image size
    args.image_size = [int(img_size) for img_size in args.image_size.split(',')]
    if len(args.image_size) == 1:
        args.image_size = args.image_size[0]
    MyRandomResizedCrop.CONTINUOUS = args.continuous_size
    MyRandomResizedCrop.SYNC_DISTRIBUTED = not args.not_sync_distributed_image_size

    # build run config from args
    args.lr_schedule_param = None
    args.opt_param = {
        'momentum': args.momentum,
        'nesterov': not args.no_nesterov,
    }
    args.init_lr = args.base_lr # linearly rescale the learning rate
    if args.warmup_lr < 0:
        args.warmup_lr = args.base_lr
    args.train_batch_size = args.base_batch_size
    args.test_batch_size = 1
    run_config = Div2K_SetXXRunConfig(**args.__dict__)

    # print run config information
    # if hvd.rank() == 0:
    #     print('Run config:')
    #     for k, v in run_config.config.items():
    #         print('\t%s: %s' % (k, v))

    if args.dy_conv_scaling_mode == -1:
        args.dy_conv_scaling_mode = None
    DynamicSeparableConv2d.KERNEL_TRANSFORM_MODE = args.dy_conv_scaling_mode

    # build net from args
    args.width_mult_list = [float(width_mult) for width_mult in args.width_mult_list.split(',')]
    args.ks_list = [int(ks) for ks in args.ks_list.split(',')]
    args.expand_list = [int(e) for e in args.expand_list.split(',')]
    args.depth_list = [int(d) for d in args.depth_list.split(',')]
    args.pixelshuffle_depth_list = [int(pixel_d) for pixel_d in args.pixelshuffle_depth_list.split(',')]

    net = OFAMobileNetS4(
        bn_param=(args.bn_momentum, args.bn_eps),
        dropout_rate=args.dropout, base_stage_width=args.base_stage_width, width_mult_list=args.width_mult_list,
        ks_list=args.ks_list, expand_ratio_list=args.expand_list, depth_list=args.depth_list, pixelshuffle_depth_list=args.pixelshuffle_depth_list
    )
    # teacher model
    # if args.kd_ratio > 0:
    #     args.teacher_model = OFAMobileNetV3(
    #         n_classes=run_config.data_provider.n_classes, bn_param=(args.bn_momentum, args.bn_eps),
    #         dropout_rate=0, width_mult_list=1.0, ks_list=7, expand_ratio_list=6, depth_list=4,
    #     )
    #     args.teacher_model.cuda()
    args.teacher_model = None

    """ Distributed RunManager """
    # Horovod: (optional) compression algorithm.
    # compression = hvd.Compression.fp16 if args.fp16_allreduce else hvd.Compression.none
    # distributed_run_manager = DistributedRunManager(
    #     args.path, net, run_config, compression, backward_steps=args.dynamic_batch_size, is_root=(hvd.rank() == 0)
    # )
    run_manager = SRRunManager(
        args.path, net, run_config, num_gpus=args.num_gpus, args=args
    )
    # distributed_run_manager.save_config()
    run_manager.save_config()
    # hvd broadcast
    # distributed_run_manager.broadcast()

    # load teacher net weights
    # if args.kd_ratio > 0:
    #     load_models(distributed_run_manager, args.teacher_model, model_path=args.teacher_path)
    model_path = './complete/sr_x2_k5_e3_d2_bn_mse/checkpoint/model_best.pth.tar'  # 이거도 매번 바꿔줘야한다.
    load_models(run_manager, run_manager.net, model_path=model_path)
    run_manager.net.module.set_active_subnet(ks=5, e=3, d=2, pixel_d=1)

    # training
    # from ofa.elastic_nn.training.progressive_shrinking import validate, train

    # validate_func_dict = {'image_size_list': {224} if isinstance(args.image_size, int) else sorted({160, 224}),
    #                       'width_mult_list': sorted({0, len(args.width_mult_list) - 1}),
    #                       'ks_list': sorted({min(args.ks_list), max(args.ks_list)}),
    #                       'expand_ratio_list': sorted({min(args.expand_list), max(args.expand_list)}),
    #                       'depth_list': sorted({min(net.depth_list), max(net.depth_list)})}
    # if args.task == 'kernel':
    #     validate_func_dict['ks_list'] = sorted(args.ks_list)
    #     if distributed_run_manager.start_epoch == 0:
    #         model_path = download_url('https://hanlab.mit.edu/files/OnceForAll/ofa_checkpoints/ofa_D4_E6_K7',
    #                                   model_dir='.torch/ofa_checkpoints/%d' % hvd.rank())
    #         load_models(distributed_run_manager, distributed_run_manager.net, model_path=model_path)
    #         distributed_run_manager.write_log('%.3f\t%.3f\t%.3f\t%s' %
    #                                           validate(distributed_run_manager, **validate_func_dict), 'valid')
    #     train(distributed_run_manager, args,
    #           lambda _run_manager, epoch, is_test: validate(_run_manager, epoch, is_test, **validate_func_dict))
    # elif args.task == 'depth':
    #     from ofa.elastic_nn.training.progressive_shrinking import supporting_elastic_depth
    #     supporting_elastic_depth(train, distributed_run_manager, args, validate_func_dict)
    # else:
    #     from ofa.elastic_nn.training.progressive_shrinking import supporting_elastic_expand
    #     supporting_elastic_expand(train, distributed_run_manager, args, validate_func_dict)

    # valid before train
    run_manager.validate()
    # run_manager.train(args)
    # run_manager.validate(tensorboard_logging=True)