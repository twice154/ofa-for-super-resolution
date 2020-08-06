# Once for All: Train One Network and Specialize it for Efficient Deployment
# Han Cai, Chuang Gan, Tianzhe Wang, Zhekai Zhang, Song Han
# International Conference on Learning Representations (ICLR), 2020.

# from imagenet_codebase.run_manager.run_manager import *
# from imagenet_codebase.data_providers.imagenet import *
from ..data_providers.imagenet import ImagenetDataProvider
from ..data_providers.cifar10 import Cifar10DataProvider
from ..data_providers.div2k_setxx import Div2K_SetXXDataProvider
from ..data_providers.oracle_video import Oracle_VideoDataProvider

from .run_manager import RunConfig
from .run_manager import RunManager
# from .distributed_run_manager import DistributedMetric, DistributedRunManager


class ImagenetRunConfig(RunConfig):

    def __init__(self, n_epochs=150, init_lr=0.05, lr_schedule_type='cosine', lr_schedule_param=None,
                 dataset='imagenet', train_batch_size=256, test_batch_size=500, valid_size=None,
                 opt_type='sgd', opt_param=None, weight_decay=4e-5, label_smoothing=0.1, no_decay_keys=None,
                 mixup_alpha=None,
                 model_init='he_fout', validation_frequency=1, print_frequency=10,
                 n_worker=32, resize_scale=0.08, distort_color='tf', image_size=224, **kwargs):
        super(ImagenetRunConfig, self).__init__(
            n_epochs, init_lr, lr_schedule_type, lr_schedule_param,
            dataset, train_batch_size, test_batch_size, valid_size,
            opt_type, opt_param, weight_decay, label_smoothing, no_decay_keys,
            mixup_alpha,
            model_init, validation_frequency, print_frequency
        )

        self.n_worker = n_worker
        self.resize_scale = resize_scale
        self.distort_color = distort_color
        self.image_size = image_size

    @property
    def data_provider(self):
        if self.__dict__.get('_data_provider', None) is None:
            if self.dataset == ImagenetDataProvider.name():
                DataProviderClass = ImagenetDataProvider
            else:
                raise NotImplementedError
            self.__dict__['_data_provider'] = DataProviderClass(
                train_batch_size=self.train_batch_size, test_batch_size=self.test_batch_size,
                valid_size=self.valid_size, n_worker=self.n_worker, resize_scale=self.resize_scale,
                distort_color=self.distort_color, image_size=self.image_size,
            )
        return self.__dict__['_data_provider']


class DistributedImageNetRunConfig(ImagenetRunConfig):

    def __init__(self, n_epochs=150, init_lr=0.05, lr_schedule_type='cosine', lr_schedule_param=None,
                 dataset='imagenet', train_batch_size=64, test_batch_size=64, valid_size=None,
                 opt_type='sgd', opt_param=None, weight_decay=4e-5, label_smoothing=0.1, no_decay_keys=None,
                 mixup_alpha=None,
                 model_init='he_fout', validation_frequency=1, print_frequency=10,
                 n_worker=8, resize_scale=0.08, distort_color='tf', image_size=224,
                 **kwargs):
        super(DistributedImageNetRunConfig, self).__init__(
            n_epochs, init_lr, lr_schedule_type, lr_schedule_param,
            dataset, train_batch_size, test_batch_size, valid_size,
            opt_type, opt_param, weight_decay, label_smoothing, no_decay_keys,
            mixup_alpha,
            model_init, validation_frequency, print_frequency, n_worker, resize_scale, distort_color, image_size,
            **kwargs
        )

        self._num_replicas = kwargs['num_replicas']
        self._rank = kwargs['rank']

    @property
    def data_provider(self):
        if self.__dict__.get('_data_provider', None) is None:
            if self.dataset == ImagenetDataProvider.name():
                DataProviderClass = ImagenetDataProvider
            else:
                raise NotImplementedError
            self.__dict__['_data_provider'] = DataProviderClass(
                train_batch_size=self.train_batch_size, test_batch_size=self.test_batch_size,
                valid_size=self.valid_size, n_worker=self.n_worker, resize_scale=self.resize_scale,
                distort_color=self.distort_color, image_size=self.image_size,
                num_replicas=self._num_replicas, rank=self._rank,
            )
        return self.__dict__['_data_provider']


class Cifar10RunConfig(RunConfig):

    def __init__(self, n_epochs=150, init_lr=0.05, lr_schedule_type='cosine', lr_schedule_param=None,
                 dataset='cifar10', train_batch_size=256, test_batch_size=500, valid_size=None,
                 opt_type='sgd', opt_param=None, weight_decay=4e-5, label_smoothing=0.1, no_decay_keys=None,
                 mixup_alpha=None,
                 model_init='he_fout', validation_frequency=1, print_frequency=10,
                 n_worker=32, resize_scale=0.08, distort_color=None, image_size=32, **kwargs):
        super(Cifar10RunConfig, self).__init__(
            n_epochs, init_lr, lr_schedule_type, lr_schedule_param,
            dataset, train_batch_size, test_batch_size, valid_size,
            opt_type, opt_param, weight_decay, label_smoothing, no_decay_keys,
            mixup_alpha,
            model_init, validation_frequency, print_frequency
        )

        self.n_worker = n_worker
        self.resize_scale = resize_scale
        self.distort_color = distort_color
        self.image_size = image_size

    @property
    def data_provider(self):
        if self.__dict__.get('_data_provider', None) is None:
            if self.dataset == Cifar10DataProvider.name():
                DataProviderClass = Cifar10DataProvider
            else:
                raise NotImplementedError
            self.__dict__['_data_provider'] = DataProviderClass(
                train_batch_size=self.train_batch_size, test_batch_size=self.test_batch_size,
                valid_size=self.valid_size, n_worker=self.n_worker, resize_scale=self.resize_scale,
                distort_color=self.distort_color, image_size=self.image_size,
            )
        return self.__dict__['_data_provider']


class Div2K_SetXXRunConfig(RunConfig):

    def __init__(self, n_epochs=150, init_lr=0.05, lr_schedule_type='cosine', lr_schedule_param=None,
                 dataset='div2k_setxx', train_batch_size=256, test_batch_size=500, valid_size=None,
                 opt_type='sgd', opt_param=None, weight_decay=4e-5, label_smoothing=0.1, no_decay_keys=None,
                 mixup_alpha=None,
                 model_init='he_fout', validation_frequency=1, print_frequency=10,
                 n_worker=32, resize_scale=0.08, distort_color=None, image_size=32, **kwargs):
        super(Div2K_SetXXRunConfig, self).__init__(
            n_epochs, init_lr, lr_schedule_type, lr_schedule_param,
            dataset, train_batch_size, test_batch_size, valid_size,
            opt_type, opt_param, weight_decay, label_smoothing, no_decay_keys,
            mixup_alpha,
            model_init, validation_frequency, print_frequency
        )

        self.n_worker = n_worker
        self.resize_scale = resize_scale
        self.distort_color = distort_color
        self.image_size = image_size

    @property
    def data_provider(self):
        if self.__dict__.get('_data_provider', None) is None:
            if self.dataset == Div2K_SetXXDataProvider.name():
                DataProviderClass = Div2K_SetXXDataProvider
            else:
                raise NotImplementedError
            self.__dict__['_data_provider'] = DataProviderClass(
                train_batch_size=self.train_batch_size, test_batch_size=self.test_batch_size,
                valid_size=self.valid_size, n_worker=self.n_worker, resize_scale=self.resize_scale,
                distort_color=self.distort_color, image_size=self.image_size,
            )
        return self.__dict__['_data_provider']


class Oracle_VideoRunConfig(RunConfig):

    def __init__(self, n_epochs=150, init_lr=0.05, lr_schedule_type='cosine', lr_schedule_param=None,
                 dataset='oracle_video', train_batch_size=256, test_batch_size=500, valid_size=None,
                 opt_type='sgd', opt_param=None, weight_decay=4e-5, label_smoothing=0.1, no_decay_keys=None,
                 mixup_alpha=None,
                 model_init='he_fout', validation_frequency=1, print_frequency=10,
                 n_worker=32, resize_scale=0.08, distort_color=None, image_size=32, **kwargs):
        super(Oracle_VideoRunConfig, self).__init__(
            n_epochs, init_lr, lr_schedule_type, lr_schedule_param,
            dataset, train_batch_size, test_batch_size, valid_size,
            opt_type, opt_param, weight_decay, label_smoothing, no_decay_keys,
            mixup_alpha,
            model_init, validation_frequency, print_frequency
        )

        self.n_worker = n_worker
        self.resize_scale = resize_scale
        self.distort_color = distort_color
        self.image_size = image_size

    @property
    def data_provider(self):
        if self.__dict__.get('_data_provider', None) is None:
            if self.dataset == Oracle_VideoDataProvider.name():
                DataProviderClass = Oracle_VideoDataProvider
            else:
                raise NotImplementedError
            self.__dict__['_data_provider'] = DataProviderClass(
                train_batch_size=self.train_batch_size, test_batch_size=self.test_batch_size,
                valid_size=self.valid_size, n_worker=self.n_worker, resize_scale=self.resize_scale,
                distort_color=self.distort_color, image_size=self.image_size,
            )
        return self.__dict__['_data_provider']