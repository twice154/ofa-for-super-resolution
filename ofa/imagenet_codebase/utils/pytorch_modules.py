# Once for All: Train One Network and Specialize it for Efficient Deployment
# Han Cai, Chuang Gan, Tianzhe Wang, Zhekai Zhang, Song Han
# International Conference on Learning Representations (ICLR), 2020.

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F


def make_divisible(v, divisor, min_val=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_val:
    :return:
    """
    if min_val is None:
        min_val = divisor
    new_v = max(min_val, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def build_activation(act_func, inplace=True, upscale_factor=2):
    if act_func == 'relu':
        return nn.ReLU(inplace=inplace)
    elif act_func == 'relu6':
        return nn.ReLU6(inplace=inplace)
    elif act_func == 'tanh':
        return nn.Tanh()
    elif act_func == 'sigmoid':
        return nn.Sigmoid()
    elif act_func == 'h_swish':
        return Hswish(inplace=inplace)
    elif act_func == 'h_sigmoid':
        return Hsigmoid(inplace=inplace)
    elif act_func == 'prelu':
        return nn.PReLU(inplace=inplace)
    elif act_func == 'lrelu':
        return nn.LeakyReLU(0.1, inplace=inplace)
    elif act_func == 'pixelshuffle':
        return build_pixelshuffle(upscale_factor=2)
    elif act_func == 'pixelshuffle+relu':
        return nn.Sequential(
            build_pixelshuffle(upscale_factor=upscale_factor),
            nn.ReLU(inplace=inplace)
        )
    elif act_func == 'pixelshuffle+relu6':
        return nn.Sequential(
            build_pixelshuffle(upscale_factor=upscale_factor),
            nn.ReLU6(inplace=inplace)
        )
    elif act_func == 'pixelshuffle+prelu':
        return nn.Sequential(
            build_pixelshuffle(upscale_factor=upscale_factor),
            nn.PReLU(inplace=inplace)
        )
    elif act_func == 'pixelshuffle+lrelu':
        return nn.Sequential(
            build_pixelshuffle(upscale_factor=upscale_factor),
            nn.LeakyReLU(0.1, inplace=inplace)
        )
    elif act_func == 'pixelunshuffle':
        return build_pixelunshuffle(downscale_factor=2)
    elif act_func == 'pixelunshuffle+relu':
        return nn.Sequential(
            build_pixelunshuffle(downscale_factor=upscale_factor),
            nn.ReLU(inplace=inplace)
        )
    elif act_func == 'pixelunshuffle+relu6':
        return nn.Sequential(
            build_pixelunshuffle(downscale_factor=upscale_factor),
            nn.ReLU6(inplace=inplace)
        )
    elif act_func == 'pixelunshuffle+prelu':
        return nn.Sequential(
            build_pixelunshuffle(downscale_factor=upscale_factor),
            nn.PReLU(inplace=inplace)
        )
    elif act_func == 'pixelunshuffle+lrelu':
        return nn.Sequential(
            build_pixelunshuffle(downscale_factor=upscale_factor),
            nn.LeakyReLU(0.1, inplace=inplace)
        )
    elif act_func is None:
        return None
    else:
        raise ValueError('do not support: %s' % act_func)


def build_pixelshuffle(upscale_factor=2):
    return nn.PixelShuffle(upscale_factor=upscale_factor)


def build_pixelunshuffle(downscale_factor=2):
    return PixelUnshuffle(downscale_factor=downscale_factor)
    

class ShuffleLayer(nn.Module):
    
    def __init__(self, groups):
        super(ShuffleLayer, self).__init__()
        self.groups = groups
    
    def forward(self, x):
        batchsize, num_channels, height, width = x.size()
        channels_per_group = num_channels // self.groups
        # reshape
        x = x.view(batchsize, self.groups, channels_per_group, height, width)
        x = torch.transpose(x, 1, 2).contiguous()
        # flatten
        x = x.view(batchsize, -1, height, width)
        return x


class Hswish(nn.Module):
    
    def __init__(self, inplace=True):
        super(Hswish, self).__init__()
        self.inplace = inplace
    
    def forward(self, x):
        return x * F.relu6(x + 3., inplace=self.inplace) / 6.


class Hsigmoid(nn.Module):
    
    def __init__(self, inplace=True):
        super(Hsigmoid, self).__init__()
        self.inplace = inplace
    
    def forward(self, x):
        return F.relu6(x + 3., inplace=self.inplace) / 6.


class SEModule(nn.Module):
    REDUCTION = 4
    
    def __init__(self, channel):
        super(SEModule, self).__init__()
        
        self.channel = channel
        self.reduction = SEModule.REDUCTION

        num_mid = make_divisible(self.channel // self.reduction, divisor=8)
        
        self.fc = nn.Sequential(OrderedDict([
            ('reduce', nn.Conv2d(self.channel, num_mid, 1, 1, 0, bias=True)),
            ('relu', nn.ReLU(inplace=True)),
            ('expand', nn.Conv2d(num_mid, self.channel, 1, 1, 0, bias=True)),
            ('h_sigmoid', Hsigmoid(inplace=True)),
        ]))
    
    def forward(self, x):
        y = x.mean(3, keepdim=True).mean(2, keepdim=True)
        y = self.fc(y)
        return x * y


# import torch
# import torch.nn as nn
# import torch.nn.functional as F


def pixel_unshuffle(input, downscale_factor):
    '''
    input: batchSize * c * k*w * k*h
    kdownscale_factor: k
    batchSize * c * k*w * k*h -> batchSize * k*k*c * w * h
    '''
    c = input.shape[1]

    kernel = torch.zeros(size=[downscale_factor * downscale_factor * c,
                               1, downscale_factor, downscale_factor],
                         device=input.device)
    for y in range(downscale_factor):
        for x in range(downscale_factor):
            kernel[x + y * downscale_factor::downscale_factor*downscale_factor, 0, y, x] = 1
    return F.conv2d(input, kernel, stride=downscale_factor, groups=c)

class PixelUnshuffle(nn.Module):
    def __init__(self, downscale_factor):
        super(PixelUnshuffle, self).__init__()
        self.downscale_factor = downscale_factor
    def forward(self, input):
        '''
        input: batchSize * c * k*w * k*h
        kdownscale_factor: k
        batchSize * c * k*w * k*h -> batchSize * k*k*c * w * h
        '''

        return pixel_unshuffle(input, self.downscale_factor)