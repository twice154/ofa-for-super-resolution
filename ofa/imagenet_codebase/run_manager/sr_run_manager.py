# Once for All: Train One Network and Specialize it for Efficient Deployment
# Han Cai, Chuang Gan, Tianzhe Wang, Zhekai Zhang, Song Han
# International Conference on Learning Representations (ICLR), 2020.

import os
import time
import json
import math
from tqdm import tqdm

import numpy as np

import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torchvision

# from imagenet_codebase.utils import *
from ..utils import get_net_info, cross_entropy_loss_with_soft_target, cross_entropy_with_label_smoothing
from ofa.utils import  AverageMeter, accuracy, psnr


class RunConfig:

    def __init__(self, n_epochs, init_lr, lr_schedule_type, lr_schedule_param,
                 dataset, train_batch_size, test_batch_size, valid_size,
                 opt_type, opt_param, weight_decay, label_smoothing, no_decay_keys,
                 mixup_alpha,
                 model_init, validation_frequency, print_frequency):
        self.n_epochs = n_epochs
        self.init_lr = init_lr
        self.lr_schedule_type = lr_schedule_type
        self.lr_schedule_param = lr_schedule_param

        self.dataset = dataset
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.valid_size = valid_size

        self.opt_type = opt_type
        self.opt_param = opt_param
        self.weight_decay = weight_decay
        self.label_smoothing = label_smoothing
        self.no_decay_keys = no_decay_keys

        self.mixup_alpha = mixup_alpha

        self.model_init = model_init
        self.validation_frequency = validation_frequency
        self.print_frequency = print_frequency

    @property
    def config(self):
        config = {}
        for key in self.__dict__:
            if not key.startswith('_'):
                config[key] = self.__dict__[key]
        return config

    def copy(self):
        return RunConfig(**self.config)

    """ learning rate """

    def calc_learning_rate(self, epoch, batch=0, nBatch=None):
        if self.lr_schedule_type == 'cosine':
            T_total = self.n_epochs * nBatch
            T_cur = epoch * nBatch + batch
            lr = 0.5 * self.init_lr * (1 + math.cos(math.pi * T_cur / T_total))
        elif self.lr_schedule_type is None:
            lr = self.init_lr
        else:
            raise ValueError('do not support: %s' % self.lr_schedule_type)
        return lr

    def adjust_learning_rate(self, optimizer, epoch, batch=0, nBatch=None):
        """ adjust learning of a given optimizer and return the new learning rate """
        new_lr = self.calc_learning_rate(epoch, batch, nBatch)
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr
        return new_lr

    def warmup_adjust_learning_rate(self, optimizer, T_total, nBatch, epoch, batch=0, warmup_lr=0):
        T_cur = epoch * nBatch + batch + 1
        new_lr = T_cur / T_total * (self.init_lr - warmup_lr) + warmup_lr
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr
        return new_lr

    """ data provider """

    @property
    def data_provider(self):
        raise NotImplementedError

    @property
    def train_loader(self):
        return self.data_provider.train

    @property
    def valid_loader(self):
        return self.data_provider.valid

    @property
    def test_loader(self):
        return self.data_provider.test

    def random_sub_train_loader(self, n_images, batch_size, num_worker=None, num_replicas=None, rank=None):
        return self.data_provider.build_sub_train_loader(n_images, batch_size, num_worker, num_replicas, rank)

    """ optimizer """

    def build_optimizer(self, net_params):
        if self.no_decay_keys is not None:
            assert isinstance(net_params, list) and len(net_params) == 2
            net_params = [
                {'params': net_params[0], 'weight_decay': self.weight_decay},
                {'params': net_params[1], 'weight_decay': 0},
            ]
        else:
            net_params = [{'params': net_params, 'weight_decay': self.weight_decay}]

        if self.opt_type == 'sgd':
            opt_param = {} if self.opt_param is None else self.opt_param
            momentum, nesterov = opt_param.get('momentum', 0.9), opt_param.get('nesterov', True)
            optimizer = torch.optim.SGD(net_params, self.init_lr, momentum=momentum, nesterov=nesterov)
        elif self.opt_type == 'adam':
            optimizer = torch.optim.Adam(net_params, self.init_lr)
        else:
            raise NotImplementedError
        return optimizer


class SRRunManager:

    def __init__(self, path, net, run_config: RunConfig, init=True, measure_latency=None, no_gpu=False, mix_prec=None, num_gpus=None, args=None):
        self.path = path
        self.net = net
        self.run_config = run_config
        self.mix_prec = mix_prec

        self.best_acc = 0
        self.start_epoch = 0

        os.makedirs(self.path, exist_ok=True)

        # move network to GPU if available
        if torch.cuda.is_available() and (not no_gpu):
            self.device = torch.device('cuda:0')
            self.net = self.net.to(self.device)
            cudnn.benchmark = True
        else:
            self.device = torch.device('cpu')
        # initialize model (default)
        if init:
            self.network.init_model(run_config.model_init)

        # net info
        net_info = get_net_info(self.net, self.run_config.data_provider.data_shape, measure_latency, True, args)
        with open('%s/net_info.txt' % self.path, 'w') as fout:
            fout.write(json.dumps(net_info, indent=4) + '\n')
            try:
                fout.write(self.network.module_str)
            except Exception:
                pass

        # criterion
        if isinstance(self.run_config.mixup_alpha, float):
            self.train_criterion = cross_entropy_loss_with_soft_target
        elif self.run_config.label_smoothing > 0:
            self.train_criterion = lambda pred, target: \
                cross_entropy_with_label_smoothing(pred, target, self.run_config.label_smoothing)
        else:
            self.train_criterion = nn.MSELoss()  # nn.L1Loss()
        self.test_criterion = nn.MSELoss()  # nn.L1Loss()

        # optimizer
        if self.run_config.no_decay_keys:
            keys = self.run_config.no_decay_keys.split('#')
            net_params = [
                self.network.get_parameters(keys, mode='exclude'),  # parameters with weight decay
                self.network.get_parameters(keys, mode='include'),  # parameters without weight decay
            ]
        else:
            try:
                net_params = self.network.weight_parameters()
            except Exception:
                net_params = self.network.parameters()
        self.optimizer = self.run_config.build_optimizer(net_params)

        if mix_prec is not None:
            from apex import amp
            self.network, self.optimizer = amp.initialize(self.network, self.optimizer, opt_level=mix_prec)

        if num_gpus > 1:
            self.net = torch.nn.DataParallel(self.net)

    """ save path and log path """

    @property
    def save_path(self):
        if self.__dict__.get('_save_path', None) is None:
            save_path = os.path.join(self.path, 'checkpoint')
            os.makedirs(save_path, exist_ok=True)
            self.__dict__['_save_path'] = save_path
        return self.__dict__['_save_path']

    @property
    def logs_path(self):
        if self.__dict__.get('_logs_path', None) is None:
            logs_path = os.path.join(self.path, 'logs')
            os.makedirs(logs_path, exist_ok=True)
            self.__dict__['_logs_path'] = logs_path
        return self.__dict__['_logs_path']

    @property
    def network(self):
        if isinstance(self.net, nn.DataParallel):
            return self.net.module
        else:
            return self.net

    @network.setter
    def network(self, new_val):
        if isinstance(self.net, nn.DataParallel):
            self.net.module = new_val
        else:
            self.net = new_val

    def write_log(self, log_str, prefix='valid', should_print=True):
        """ prefix: valid, train, test """
        if prefix in ['valid', 'test']:
            with open(os.path.join(self.logs_path, 'valid_console.txt'), 'a') as fout:
                fout.write(log_str + '\n')
                fout.flush()
        if prefix in ['valid', 'test', 'train']:
            with open(os.path.join(self.logs_path, 'train_console.txt'), 'a') as fout:
                if prefix in ['valid', 'test']:
                    fout.write('=' * 10)
                fout.write(log_str + '\n')
                fout.flush()
        else:
            with open(os.path.join(self.logs_path, '%s.txt' % prefix), 'a') as fout:
                fout.write(log_str + '\n')
                fout.flush()
        if should_print:
            print(log_str)

    """ save and load models """

    def save_model(self, checkpoint=None, is_best=False, model_name=None):
        if checkpoint is None:
            checkpoint = {'state_dict': self.network.state_dict()}

        if model_name is None:
            model_name = 'checkpoint.pth.tar'

        if self.mix_prec is not None:
            from apex import amp
            checkpoint['amp'] = amp.state_dict()

        checkpoint['dataset'] = self.run_config.dataset  # add `dataset` info to the checkpoint
        latest_fname = os.path.join(self.save_path, 'latest.txt')
        model_path = os.path.join(self.save_path, model_name)
        with open(latest_fname, 'w') as fout:
            fout.write(model_path + '\n')
        torch.save(checkpoint, model_path)

        if is_best:
            best_path = os.path.join(self.save_path, 'model_best.pth.tar')
            torch.save({'state_dict': checkpoint['state_dict']}, best_path)

    def load_model(self, model_fname=None):
        latest_fname = os.path.join(self.save_path, 'latest.txt')
        if model_fname is None and os.path.exists(latest_fname):
            with open(latest_fname, 'r') as fin:
                model_fname = fin.readline()
                if model_fname[-1] == '\n':
                    model_fname = model_fname[:-1]
        try:
            if model_fname is None or not os.path.exists(model_fname):
                model_fname = '%s/checkpoint.pth.tar' % self.save_path
                with open(latest_fname, 'w') as fout:
                    fout.write(model_fname + '\n')
            print("=> loading checkpoint '{}'".format(model_fname))

            if torch.cuda.is_available():
                checkpoint = torch.load(model_fname)
            else:
                checkpoint = torch.load(model_fname, map_location='cpu')

            self.network.load_state_dict(checkpoint['state_dict'])

            if 'epoch' in checkpoint:
                self.start_epoch = checkpoint['epoch'] + 1
            if 'best_acc' in checkpoint:
                self.best_acc = checkpoint['best_acc']
            if 'optimizer' in checkpoint:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            if self.mix_prec is not None and 'amp' in checkpoint:
                from apex import amp
                amp.load_state_dict(checkpoint['amp'])

            print("=> loaded checkpoint '{}'".format(model_fname))
        except Exception:
            print('fail to load checkpoint from %s' % self.save_path)

    def save_config(self):
        """ dump run_config and net_config to the model_folder """
        #################### 이부분이 아마 down-image guide 세팅으로 실험할 때, 에러나는 부분임. 그게 아니라 그냥 에러나니까 별거 아닌 것 같으니 주석하고 쓰자.
        # net_save_path = os.path.join(self.path, 'net.config')
        # json.dump(self.network.config, open(net_save_path, 'w'), indent=4)
        # print('Network configs dump to %s' % net_save_path)

        run_save_path = os.path.join(self.path, 'run.config')
        json.dump(self.run_config.config, open(run_save_path, 'w'), indent=4)
        print('Run configs dump to %s' % run_save_path)

    """ train and test """

    def validate(self, epoch=0, is_test=True, run_str='', net=None, data_loader=None, no_logs=False, tensorboard_logging=False):
        if tensorboard_logging:
            from tensorboardX import SummaryWriter  ################## for tensorboardX. Seuqential Video에 대해서 로그찍을 필요 없을때는 그냥 삭제하면됨.
            writer = SummaryWriter('./runs/sr_teacher_bn_mse_bolt')  ################## 필요할 때마다 log위치 수정가능. for tensorboardX. Seuqential Video에 대해서 로그찍을 필요 없을때는 그냥 삭제하면됨.
            
        if net is None:
            net = self.net
        if not isinstance(net, nn.DataParallel):
            net = nn.DataParallel(net)

        if data_loader is None:
            if is_test:
                data_loader = self.run_config.test_loader
            else:
                data_loader = self.run_config.valid_loader

        net.eval()

        losses = AverageMeter()
        # top1 = AverageMeter()
        # top5 = AverageMeter()
        psnr_averagemeter = AverageMeter()

        with torch.no_grad():
            with tqdm(total=len(data_loader),
                      desc='Validate Epoch #{} {}'.format(epoch + 1, run_str), disable=no_logs) as t:
                for i, mini_batch in enumerate(data_loader):
                    images = mini_batch['image']
                    #################### 2x or 4x 고르는 부분.
                    x2_down_images = mini_batch['2x_down_image']
                    # x4_down_images = mini_batch['4x_down_image']
                    images = images.to(self.device)
                    #################### 2x or 4x 고르는 부분.
                    x2_down_images = x2_down_images.to(self.device)
                    # x4_down_images = x4_down_images.to(self.device)
                    # compute output
                    #################### 2x or 4x 고르는 부분.
                    output = net(x2_down_images)
                    # output = net(x4_down_images)
                    loss = self.test_criterion(output, images)
                    # measure accuracy and record loss
                    psnr_current = psnr(rgb2y(tensor2img_np(output)), rgb2y(tensor2img_np(images)))  # HR Comparison
                    # import PIL  # LR Comparison
                    # import torchvision.transforms as transforms  # LR Comparison
                    # output = output.cpu().data[0, :, :, :]  # LR Comparison
                    # output = transforms.ToPILImage()(output)  # LR Comparison
                    # output = output.resize((int(output.size[0]/2), int(output.size[1]/2)), resample=PIL.Image.BICUBIC)  # LR Comparison
                    # output.save('zssr.png')  # LR Comparison FOR VALIDATE BICUBIC_DOWN
                    # output = transforms.ToTensor()(output)  # LR Comparison
                    # psnr_current = psnr(rgb2y(tensor2img_np(output)), rgb2y(tensor2img_np(x2_down_images)))  # LR Comparison
                    
                    if tensorboard_logging:
                        writer.add_scalars('metric', {'psnr': psnr_current}, i)  ################## for tensorboardX. Seuqential Video에 대해서 로그찍을 필요 없을때는 그냥 삭제하면됨.

                    losses.update(loss.item(), images.size(0))
                    # top1.update(acc1[0].item(), images.size(0))
                    # top5.update(acc5[0].item(), images.size(0))
                    psnr_averagemeter.update(psnr_current, images.size(0))
                    t.set_postfix({
                        'loss': losses.avg,
                        # 'top1': top1.avg,
                        # 'top5': top5.avg,
                        'psnr': psnr_averagemeter.avg,
                        'img_size': images.size(2),
                    })
                    t.update(1)

        if tensorboard_logging:
            writer.close()  #################### for tensorboardX. Seuqential Video에 대해서 로그찍을 필요 없을때는 그냥 삭제하면됨.

        return losses.avg, psnr_averagemeter.avg

    # def validate_all_resolution(self, epoch=0, is_test=True, net=None):
    #     if net is None:
    #         net = self.network
    #     if isinstance(self.run_config.data_provider.image_size, list):
    #         img_size_list, loss_list, top1_list, top5_list = [], [], [], []
    #         for img_size in self.run_config.data_provider.image_size:
    #             img_size_list.append(img_size)
    #             self.run_config.data_provider.assign_active_img_size(img_size)
    #             self.reset_running_statistics(net=net)
    #             loss, top1, top5 = self.validate(epoch, is_test, net=net)
    #             loss_list.append(loss)
    #             top1_list.append(top1)
    #             top5_list.append(top5)
    #         return img_size_list, loss_list, top1_list, top5_list
    #     else:
    #         loss, top1, top5 = self.validate(epoch, is_test, net=net)
    #         return [self.run_config.data_provider.active_img_size], [loss], [top1], [top5]

    def train_one_epoch(self, args, epoch, warmup_epochs=0, warmup_lr=0):
        # switch to train mode
        self.net.train()
        #################### Code for freezing BN. Overfitting 할 때는 주석 해제하면됨.
        for m in self.net.modules():
            if isinstance(m, nn.BatchNorm2d):
                ########## Use running mean/var
                m.eval()
                ########## BN weight/bias freeze
                # m.weight.requires_grad = False
                # m.bias.requires_grad = False

        nBatch = len(self.run_config.train_loader)

        losses = AverageMeter()
        # top1 = AverageMeter()
        # top5 = AverageMeter()
        psnr_averagemeter = AverageMeter()
        data_time = AverageMeter()

        with tqdm(total=nBatch,
                  desc='Train Epoch #{}'.format(epoch + 1)) as t:
            end = time.time()
            for i, mini_batch in enumerate(self.run_config.train_loader):
                images = mini_batch['image']
                #################### 2x or 4x 고르는 부분.
                x2_down_images = mini_batch['2x_down_image']
                # x4_down_images = mini_batch['4x_down_image']
                data_time.update(time.time() - end)
                if epoch < warmup_epochs:
                    new_lr = self.run_config.warmup_adjust_learning_rate(
                        self.optimizer, warmup_epochs * nBatch, nBatch, epoch, i, warmup_lr,
                    )
                else:
                    new_lr = self.run_config.adjust_learning_rate(self.optimizer, epoch - warmup_epochs, i, nBatch)

                images = images.to(self.device)
                #################### 2x or 4x 고르는 부분.
                x2_down_images = x2_down_images.to(self.device)
                # x4_down_images = x4_down_images.to(self.device)
                target = images

                # soft target
                if args.teacher_model is not None:
                    args.teacher_model.train()
                    with torch.no_grad():
                        soft_logits = args.teacher_model(images).detach()
                        # soft_label = F.softmax(soft_logits, dim=1)

                # compute output
                if isinstance(self.network, torchvision.models.Inception3):
                    output, aux_outputs = self.net(images)
                    loss1 = self.train_criterion(output, labels)
                    loss2 = self.train_criterion(aux_outputs, labels)
                    loss = loss1 + 0.4 * loss2
                else:
                    #################### 2x or 4x 고르는 부분.
                    output = self.net(x2_down_images)
                    # output = self.net(x4_down_images)
                    loss = self.train_criterion(output, images)

                if args.teacher_model is None:
                    loss_type = 'mse'
                else:
                    if args.kd_type == 'ce':
                        kd_loss = cross_entropy_loss_with_soft_target(output, soft_label)
                    else:
                        kd_loss = F.mse_loss(output, soft_logits)
                    loss = args.kd_ratio * kd_loss + loss
                    loss_type = '%.1fkd-%s & mse' % (args.kd_ratio, args.kd_type)

                # compute gradient and do SGD step
                self.net.zero_grad()  # or self.optimizer.zero_grad()
                if self.mix_prec is not None:
                    from apex import amp
                    with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()
                self.optimizer.step()

                # measure accuracy and record loss
                # acc1, acc5 = accuracy(output, target, topk=(1, 5))
                psnr_current = psnr(rgb2y(tensor2img_np(output)), rgb2y(tensor2img_np(images)))
                losses.update(loss.item(), images.size(0))
                # top1.update(acc1[0].item(), images.size(0))
                # top5.update(acc5[0].item(), images.size(0))
                psnr_averagemeter.update(psnr_current, images.size(0))

                t.set_postfix({
                    'loss': losses.avg,
                    # 'top1': top1.avg,
                    # 'top5': top5.avg,
                    'psnr': psnr_averagemeter.avg,
                    'img_size': images.size(2),
                    'lr': new_lr,
                    'loss_type': loss_type,
                    'data_time': data_time.avg,
                })
                t.update(1)
                end = time.time()
        return losses.avg, psnr_averagemeter.avg

    def train(self, args, warmup_epoch=0, warmup_lr=0):
        for epoch in range(self.start_epoch, self.run_config.n_epochs + warmup_epoch):
            train_loss, train_acc = self.train_one_epoch(args, epoch, warmup_epoch, warmup_lr)  #################### naming convention상 train_psnr이라고 해야하는데, 일일이 바꾸기 귀찮아서 그냥 train_acc이라고 그대로 사용

            if (epoch + 1) % self.run_config.validation_frequency == 0:
                val_loss, val_acc = self.validate(epoch=epoch, is_test=False)  #################### naming convention상 val_psnr이라고 해야하는데, 일일이 바꾸기 귀찮아서 그냥 val_acc이라고 그대로 사용

                is_best = np.mean(val_acc) > self.best_acc   #################### naming convention상 best_psnr이라고 해야하는데, 일일이 바꾸기 귀찮아서 그냥 best_acc이라고 그대로 사용
                self.best_acc = max(self.best_acc, np.mean(val_acc))
                val_log = 'Valid [{0}/{1}]\tloss {2:.3f}\ttop-1 acc {3:.3f} ({4:.3f})'. \
                    format(epoch + 1 - warmup_epoch, self.run_config.n_epochs,
                           np.mean(val_loss), np.mean(val_acc), self.best_acc)
                val_log += '\tTrain top-1 {top1:.3f}\tloss {train_loss:.3f}\t'. \
                    format(top1=train_acc, train_loss=train_loss)
                # for i_s, v_a in zip(img_size, val_acc):
                #     val_log += '(%d, %.3f), ' % (i_s, v_a)
                self.write_log(val_log, prefix='valid', should_print=False)
            else:
                is_best = False

            self.save_model({
                'epoch': epoch,
                'best_acc': self.best_acc,
                'optimizer': self.optimizer.state_dict(),
                'state_dict': self.network.state_dict(),
            }, is_best=is_best)

    def reset_running_statistics(self, net=None):
        from ofa.elastic_nn.utils import set_running_statistics
        if net is None:
            net = self.network
        # sub_train_loader = self.run_config.random_sub_train_loader(2000, 100)
        sub_train_loader = self.run_config.train_loader
        set_running_statistics(net, sub_train_loader)


# import math
from PIL import Image
# import numpy as np
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