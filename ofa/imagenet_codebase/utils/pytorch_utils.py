# Once for All: Train One Network and Specialize it for Efficient Deployment
# Han Cai, Chuang Gan, Tianzhe Wang, Zhekai Zhang, Song Han
# International Conference on Learning Representations (ICLR), 2020.

import copy
import time

import torch
import torch.nn as nn

from ofa.imagenet_codebase.utils.flops_counter import profile


def mix_images(images, lam):
    flipped_images = torch.flip(images, dims=[0])  # flip along the batch dimension
    return lam * images + (1 - lam) * flipped_images


def mix_labels(target, lam, n_classes, label_smoothing=0.1):
    onehot_target = label_smooth(target, n_classes, label_smoothing)
    flipped_target = torch.flip(onehot_target, dims=[0])
    return lam * onehot_target + (1 - lam) * flipped_target


def label_smooth(target, n_classes: int, label_smoothing=0.1):
    # convert to one-hot
    batch_size = target.size(0)
    target = torch.unsqueeze(target, 1)
    soft_target = torch.zeros((batch_size, n_classes), device=target.device)
    soft_target.scatter_(1, target, 1)
    # label smoothing
    soft_target = soft_target * (1 - label_smoothing) + label_smoothing / n_classes
    return soft_target


def cross_entropy_loss_with_soft_target(pred, soft_target):
    logsoftmax = nn.LogSoftmax()
    return torch.mean(torch.sum(- soft_target * logsoftmax(pred), 1))


def cross_entropy_with_label_smoothing(pred, target, label_smoothing=0.1):
    soft_target = label_smooth(target, pred.size(1), label_smoothing)
    return cross_entropy_loss_with_soft_target(pred, soft_target)


def clean_num_batch_tracked(net):
    for m in net.modules():
        if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
            if m.num_batches_tracked is not None:
                m.num_batches_tracked.zero_()
                

def rm_bn_from_net(net):
    for m in net.modules():
        if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
            m.forward = lambda x: x


def module_require_grad(module):
    return module.parameters().__next__().requires_grad
            

""" Net Info """


def get_net_device(net):
    return net.parameters().__next__().device


#################### 직접 계산하는 것이 아니라, 사전 정의된 크기들의 표를 참조해서 계산하는 방식으로 간단하게 만들었다.
#################### Just Convolution 연산만... BN, SE, Activation은 포함되지 않음.
def count_parameters(net, config=None):
    if len(config.ks_list) != 1 or len(config.expand_list) != 1 or len(config.depth_list) != 1 or len(config.pixelshuffle_depth_list) != 1:
        return -1
    else:
        if config.pixelshuffle_depth_list[0] == 2:
            #################### parameter 계산에서는 width, height 계산과정에 포함안시켜면 되기때문에 그냥 1로놓고함.
            width = 1
            height = 1

            return (5 * 5 * 3 * width * height * 64) \
                    + (config.depth_list[0] * 4 * ((1 * 1 * 64 * width * height * (64 * config.expand_list[0])) + (config.ks_list[0] * config.ks_list[0] * width * height * (64 * config.expand_list[0])) + (1 * 1 * 64 * width * height * (64 * config.expand_list[0])))) \
                    + (2 * (5 * 5 * 64 * width * height * 64)) \
                    + (5 * 5 * 64 * width * height * (64 * 4)) + (5 * 5 * 64 * (width) * (height) * (64 * 4)) \
                    + (5 * 5 * 64 * (width) * (height) * 3)
        elif config.pixelshuffle_depth_list[0] == 1:
            #################### parameter 계산에서는 width, height 계산과정에 포함안시켜면 되기때문에 그냥 1로놓고함.
            width = 1
            height = 1

            return (5 * 5 * 3 * width * height * 64) \
                    + (config.depth_list[0] * 4 * ((1 * 1 * 64 * width * height * (64 * config.expand_list[0])) + (config.ks_list[0] * config.ks_list[0] * width * height * (64 * config.expand_list[0])) + (1 * 1 * 64 * width * height * (64 * config.expand_list[0])))) \
                    + (2 * (5 * 5 * 64 * width * height * 64)) \
                    + (5 * 5 * 64 * width * height * (64 * 4)) \
                    + (5 * 5 * 64 * (width) * (height) * 3)
# def count_parameters(net):
#     total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
#     return total_params


#################### 직접 계산하는 것이 아니라, 사전 정의된 크기들의 표를 참조해서 계산하는 방식으로 간단하게 만들었다.
#################### Just Convolution 연산만... BN, SE, Activation은 포함되지 않음.
def count_net_flops(net, data_shape=(1, 3, 224, 224), config=None):
    if len(config.ks_list) != 1 or len(config.expand_list) != 1 or len(config.depth_list) != 1 or len(config.pixelshuffle_depth_list) != 1:
        return -1
    else:
        if config.pixelshuffle_depth_list[0] == 2:
            width = int(data_shape[2] / 4)
            height = int(data_shape[3] / 4)

            return (5 * 5 * 3 * width * height * 64) \
                    + (config.depth_list[0] * 4 * ((1 * 1 * 64 * width * height * (64 * config.expand_list[0])) + (config.ks_list[0] * config.ks_list[0] * width * height * (64 * config.expand_list[0])) + (1 * 1 * 64 * width * height * (64 * config.expand_list[0])))) \
                    + (2 * (5 * 5 * 64 * width * height * 64)) \
                    + (5 * 5 * 64 * width * height * (64 * 4)) + (5 * 5 * 64 * (2 * width) * (2 * height) * (64 * 4)) \
                    + (5 * 5 * 64 * (4 * width) * (4 * height) * 3)
        elif config.pixelshuffle_depth_list[0] == 1:
            width = int(data_shape[2] / 2)
            height = int(data_shape[3] / 2)

            return (5 * 5 * 3 * width * height * 64) \
                    + (config.depth_list[0] * 4 * ((1 * 1 * 64 * width * height * (64 * config.expand_list[0])) + (config.ks_list[0] * config.ks_list[0] * width * height * (64 * config.expand_list[0])) + (1 * 1 * 64 * width * height * (64 * config.expand_list[0])))) \
                    + (2 * (5 * 5 * 64 * width * height * 64)) \
                    + (5 * 5 * 64 * width * height * (64 * 4)) \
                    + (5 * 5 * 64 * (2 * width) * (2 * height) * 3)
# def count_net_flops(net, data_shape=(1, 3, 224, 224)):
#     if isinstance(net, nn.DataParallel):
#         net = net.module

#     net = copy.deepcopy(net)
    
#     flop, _ = profile(net, data_shape)  # profile에서 flops 계산이 제대로 안되고 있음 (depth가 고정되어있는 일반 convolution 부분만 flops에 포함됨)
#     return flop


def measure_net_latency(net, l_type='gpu8', fast=True, input_shape=(3, 224, 224), clean=False):
    if isinstance(net, nn.DataParallel):
        net = net.module
    
    # remove bn from graph
    rm_bn_from_net(net)
    
    # return `ms`
    if 'gpu' in l_type:
        l_type, batch_size = l_type[:3], int(l_type[3:])
    else:
        batch_size = 1

    data_shape = [batch_size] + list(input_shape)
    if l_type == 'cpu':
        if fast:
            n_warmup = 5
            n_sample = 10
        else:
            n_warmup = 50
            n_sample = 50
        if get_net_device(net) != torch.device('cpu'):
            if not clean:
                print('move net to cpu for measuring cpu latency')
            net = copy.deepcopy(net).cpu()
    elif l_type == 'gpu':
        if fast:
            n_warmup = 5
            n_sample = 10
        else:
            n_warmup = 50
            n_sample = 50
    else:
        raise NotImplementedError
    images = torch.zeros(data_shape, device=get_net_device(net))

    measured_latency = {'warmup': [], 'sample': []}
    net.eval()
    with torch.no_grad():
        for i in range(n_warmup):
            inner_start_time = time.time()
            net(images)
            used_time = (time.time() - inner_start_time) * 1e3  # ms
            measured_latency['warmup'].append(used_time)
            if not clean:
                print('Warmup %d: %.3f' % (i, used_time))
        outer_start_time = time.time()
        for i in range(n_sample):
            net(images)
        total_time = (time.time() - outer_start_time) * 1e3  # ms
        measured_latency['sample'].append((total_time, n_sample))
    return total_time / n_sample, measured_latency
    

#################### input shape 부분을 네트워크에 들어가는 Image Size에 맞춰서 바꾸어주면됨. FLOPs 계산시에 이거따라서 값이 바뀜.
def get_net_info(net, input_shape=(3, 96, 96), measure_latency=None, print_info=True, config=None):
    net_info = {}
    if isinstance(net, nn.DataParallel):
        net = net.module
    
    # parameters
    net_info['params'] = count_parameters(net, config)
    
    # flops
    #################### input_shape으로 downsampled 이후의 크기가 들어오는 것이 아니라, downsampled 이전의 크기가 들어온다.
    net_info['flops'] = count_net_flops(net, [1] + list(input_shape), config)

    # storage
    net_info['weight'] = net_info['params'] * 32 / 8388608  #################### Weight가 32bit임을 가정하고 계산하는거임.
    net_info['topology'] = net_info['weight'] / 10  #################### 경험상 state_dict를 저장하기 위해 dictionary를 사용하는데, 단순 weight만 저장하는 것 보다 10% 정도 크다.
    net_info['storage'] = net_info['weight'] + net_info['topology']
    
    # latencies
    # latency_types = [] if measure_latency is None else measure_latency.split('#')
    # for l_type in latency_types:
    #     latency, measured_latency = measure_net_latency(net, l_type, fast=False, input_shape=input_shape)
    #     net_info['%s latency' % l_type] = {
    #         'val': latency,
    #         'hist': measured_latency
    #     }
    
    if print_info:
        # print(net)
        # for param_tensor in net.state_dict():
        #     print(param_tensor, "\t", net.state_dict()[param_tensor].size())
        print('Total training params: %.2fM' % (net_info['params'] / 1e6))
        print('Total FLOPs (not calculating bn flops): %.2fM' % (net_info['flops'] / 1e6))
        print('Total weight storage: %.2fMB' % (net_info['weight']))
        print('Total topology storage (slightly prediction error exist): %.2fMB' % (net_info['topology']))
        print('Total storage (slightly prediction error exist): %.2fMB' % (net_info['storage']))
        # for l_type in latency_types:
        #     print('Estimated %s latency: %.3fms' % (l_type, net_info['%s latency' % l_type]['val']))
    
    return net_info
