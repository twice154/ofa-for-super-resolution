# Once for All: Train One Network and Specialize it for Efficient Deployment
# Han Cai, Chuang Gan, Tianzhe Wang, Zhekai Zhang, Song Han
# International Conference on Learning Representations (ICLR), 2020.

import copy
import torch
import torch.nn as nn

# from layers import *
from ofa.layers import set_layer_from_config, MBInvertedConvLayer, ConvLayer, IdentityLayer, LinearLayer
from ofa.imagenet_codebase.utils import MyNetwork, make_divisible
from ofa.imagenet_codebase.networks.proxyless_nets import MobileInvertedResidualBlock


class MobileNetX4(MyNetwork):

    def __init__(self, blocks, enc_final_conv_blocks,
                dec_first_conv_block, dec_final_conv_blocks, dec_final_output_conv_block, runtime_depth):
        super(MobileNetX4, self).__init__()

        self.blocks = nn.ModuleList(blocks)
        self.enc_final_conv_blocks = nn.ModuleList(enc_final_conv_blocks)
        self.dec_first_conv_block = dec_first_conv_block
        self.dec_final_conv_blocks = nn.ModuleList(dec_final_conv_blocks)
        self.dec_final_output_conv_block = dec_final_output_conv_block

        self.runtime_depth = runtime_depth

    def forward(self, x):
        return x

    @property
    def module_str(self):
        _str = self.enc_first_conv.module_str +'\n'
        _str += self.enc_input_skip_conn.module_str + '\n'
        for block in self.enc_final_conv_blocks:
            _str += block.module_str + '\n'
        
        for block in self.blocks:
            _str += block.module_str + '\n'

        _str += self.dec_first_conv.module_str +'\n'
        _str += self.dec_input_skip_conn.module_str + '\n'
        for block in self.dec_final_conv_blocks:
            _str += block.module_str + '\n'
        
        return _str

    @property
    def config(self):
        return {
            'name': MobileNetX4.__name__,
            'bn': self.get_bn_param(),
            'enc_first_conv': self.enc_first_conv.config,
            'enc_input_skip_conn': self.enc_input_skip_conn.config,
            'enc_final_conv_blocks': [
                block.config for block in self.enc_final_conv_blocks
            ],
            'blocks': [
                block.config for block in self.blocks
            ],
            'dec_first_conv': self.dec_first_conv.config,
            'dec_input_skip_conn': self.dec_input_skip_conn.config,
            'dec_final_conv_blocks': [
                block.config for block in self.dec_final_conv_blocks
            ],
        }

    @staticmethod
    def build_from_config(config):
        enc_first_conv = set_layer_from_config(config['enc_first_conv'])
        enc_input_skip_conn = set_layer_from_config(config['enc_input_skip_conn'])
        enc_final_conv_blocks = []
        for block_config in config['enc_final_conv_blocks']:
            enc_final_conv_blocks.append(MobileInvertedResidualBlock.build_from_config(block_config))
        
        blocks = []
        for block_config in config['blocks']:
            blocks.append(MobileInvertedResidualBlock.build_from_config(block_config))

        dec_first_conv = set_layer_from_config(config['dec_first_conv'])
        dec_input_skip_conn = set_layer_from_config(config['dec_input_skip_conn'])
        dec_final_conv_blocks = []
        for block_config in config['dec_final_conv_blocks']:
            enc_final_conv_blocks.append(set_layer_from_config(block_config))
        
        net = MobileNetX4(enc_first_conv, enc_input_skip_conn, enc_final_conv_blocks,
                            blocks,
                            dec_first_conv, dec_input_skip_conn, dec_final_conv_blocks)
        if 'bn' in config:
            net.set_bn_param(**config['bn'])
        else:
            net.set_bn_param(momentum=0.1, eps=1e-3)
        
        return net

    def zero_last_gamma(self):
        for m in self.modules():
            if isinstance(m, MobileInvertedResidualBlock):
                if isinstance(m.mobile_inverted_conv, MBInvertedConvLayer) and isinstance(m.shortcut, IdentityLayer):
                    m.mobile_inverted_conv.point_linear.bn.weight.data.zero_()

#################################################################################################### 안쓰는듯?
    @staticmethod
    def build_net_via_cfg(cfg, input_channel, last_channel, n_classes, dropout_rate):
        # first conv layer
        first_conv = ConvLayer(
            3, input_channel, kernel_size=3, stride=2, use_bn=True, act_func='h_swish', ops_order='weight_bn_act'
        )
        # build mobile blocks
        feature_dim = input_channel
        blocks = []
        for stage_id, block_config_list in cfg.items():
            for k, mid_channel, out_channel, use_se, act_func, stride, expand_ratio in block_config_list:
                mb_conv = MBInvertedConvLayer(
                    feature_dim, out_channel, k, stride, expand_ratio, mid_channel, act_func, use_se
                )
                if stride == 1 and out_channel == feature_dim:
                    shortcut = IdentityLayer(out_channel, out_channel)
                else:
                    shortcut = None
                blocks.append(MobileInvertedResidualBlock(mb_conv, shortcut))
                feature_dim = out_channel
        # final expand layer
        final_expand_layer = ConvLayer(
            feature_dim, feature_dim * 6, kernel_size=1, use_bn=True, act_func='h_swish', ops_order='weight_bn_act',
        )
        feature_dim = feature_dim * 6
        # feature mix layer
        feature_mix_layer = ConvLayer(
            feature_dim, last_channel, kernel_size=1, bias=False, use_bn=False, act_func='h_swish',
        )
        # classifier
        classifier = LinearLayer(last_channel, n_classes, dropout_rate=dropout_rate)

        return first_conv, blocks, final_expand_layer, feature_mix_layer, classifier

#################################################################################################### 안쓰는듯?
    @staticmethod
    def adjust_cfg(cfg, ks=None, expand_ratio=None, depth_param=None, stage_width_list=None):
        for i, (stage_id, block_config_list) in enumerate(cfg.items()):
            for block_config in block_config_list:
                if ks is not None and stage_id != '0':
                    block_config[0] = ks
                if expand_ratio is not None and stage_id != '0':
                    block_config[-1] = expand_ratio
                    block_config[1] = None
                    if stage_width_list is not None:
                        block_config[2] = stage_width_list[i]
            if depth_param is not None and stage_id != '0':
                new_block_config_list = [block_config_list[0]]
                new_block_config_list += [copy.deepcopy(block_config_list[-1]) for _ in range(depth_param - 1)]
                cfg[stage_id] = new_block_config_list
        return cfg