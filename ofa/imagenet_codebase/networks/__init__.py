# Once for All: Train One Network and Specialize it for Efficient Deployment
# Han Cai, Chuang Gan, Tianzhe Wang, Zhekai Zhang, Song Han
# International Conference on Learning Representations (ICLR), 2020.

from ofa.imagenet_codebase.networks.proxyless_nets import ProxylessNASNets
from ofa.imagenet_codebase.networks.mobilenet_v3 import MobileNetV3, MobileNetV3Large
from ofa.imagenet_codebase.networks.mobilenet_x4 import MobileNetX4


def get_net_by_name(name):
    if name == ProxylessNASNets.__name__:
        return ProxylessNASNets
    elif name == MobileNetV3.__name__:
        return MobileNetV3
    elif name == MobileNetX4.__name__:
        return MobileNetX4
    else:
        raise ValueError('unrecognized type of network: %s' % name)
