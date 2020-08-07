# Once for All: Train One Network and Specialize it for Efficient Deployment
# Han Cai, Chuang Gan, Tianzhe Wang, Zhekai Zhang, Song Han
# International Conference on Learning Representations (ICLR), 2020.

import warnings
import os
import math
import numpy as np

import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from ofa.imagenet_codebase.data_providers.base_provider import DataProvider, MyRandomResizedCrop, MyDistributedSampler


class Oracle_VideoDataProvider(DataProvider):
    DEFAULT_PATH = '/SSD/kaist_paper_video_dataset'
    
    def __init__(self, save_path=None, train_batch_size=256, test_batch_size=512, valid_size=None, n_worker=32,
                 resize_scale=0.08, distort_color=None, image_size=32,
                 num_replicas=None, rank=None):
        
        warnings.filterwarnings('ignore')
        self._save_path = save_path
        
        self.image_size = image_size  # int or list of int
        self.distort_color = distort_color
        self.resize_scale = resize_scale

        self._valid_transform_dict = {}
        if not isinstance(self.image_size, int):
            assert isinstance(self.image_size, list)
            from ofa.imagenet_codebase.data_providers.my_data_loader import MyDataLoader
            self.image_size.sort()  # e.g., 160 -> 224
            MyRandomResizedCrop.IMAGE_SIZE_LIST = self.image_size.copy()
            MyRandomResizedCrop.ACTIVE_SIZE = max(self.image_size)

            for img_size in self.image_size:
                self._valid_transform_dict[img_size] = self.build_valid_transform(img_size)
            self.active_img_size = max(self.image_size)
            valid_transforms = self._valid_transform_dict[self.active_img_size]
            train_loader_class = MyDataLoader  # randomly sample image size for each batch of training image
        else:
            self.active_img_size = self.image_size
            valid_transforms = self.build_valid_transform()
            train_loader_class = torch.utils.data.DataLoader

        train_transforms = self.build_train_transform()
        train_dataset = self.train_dataset(train_transforms)
        
        if valid_size is not None:
            if not isinstance(valid_size, int):
                assert isinstance(valid_size, float) and 0 < valid_size < 1
                valid_size = int(len(train_dataset.samples) * valid_size)
            
            valid_dataset = self.train_dataset(valid_transforms)
            train_indexes, valid_indexes = self.random_sample_valid_set(len(train_dataset.samples), valid_size)
            
            if num_replicas is not None:
                train_sampler = MyDistributedSampler(train_dataset, num_replicas, rank, np.array(train_indexes))
                valid_sampler = MyDistributedSampler(valid_dataset, num_replicas, rank, np.array(valid_indexes))
            else:
                train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_indexes)
                valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(valid_indexes)
            
            self.train = train_loader_class(
                train_dataset, batch_size=train_batch_size, sampler=train_sampler,
                num_workers=n_worker, pin_memory=True, drop_last=True,
            )
            self.valid = torch.utils.data.DataLoader(
                valid_dataset, batch_size=test_batch_size, sampler=valid_sampler,
                num_workers=n_worker, pin_memory=True, drop_last=True,
            )
        else:
            if num_replicas is not None:
                train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas, rank)
                self.train = train_loader_class(
                    train_dataset, batch_size=train_batch_size, sampler=train_sampler,
                    num_workers=n_worker, pin_memory=True, drop_last=True,
                )
            else:
                self.train = train_loader_class(
                    train_dataset, batch_size=train_batch_size, shuffle=True,
                    num_workers=n_worker, pin_memory=True, drop_last=True,
                )
            self.valid = None
        
        test_dataset = self.test_dataset(valid_transforms)
        if num_replicas is not None:
            test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset, num_replicas, rank)
            self.test = torch.utils.data.DataLoader(
                test_dataset, batch_size=test_batch_size, sampler=test_sampler, num_workers=n_worker, pin_memory=True, drop_last=True,
            )
        else:
            self.test = torch.utils.data.DataLoader(
                test_dataset, batch_size=test_batch_size, shuffle=False, num_workers=n_worker, pin_memory=True, drop_last=True,
            )
        
        if self.valid is None:
            self.valid = self.test
    
    @staticmethod
    def name():
        return 'oracle_video'
    
    @property
    def data_shape(self):
        return 3, self.active_img_size, self.active_img_size  # C, H, W
    
    @property
    def n_classes(self):
        return 1
    
    @property
    def save_path(self):
        if self._save_path is None:
            self._save_path = self.DEFAULT_PATH
        return self._save_path
    
    @property
    def data_url(self):
        raise ValueError('unable to download %s' % self.name())
    
    def train_dataset(self, _transforms):
        dataset = Oracle_VideoDataset(self.train_path, _transforms)
        return dataset
    
    def test_dataset(self, _transforms):
        dataset = Oracle_VideoDataset(self.valid_path, _transforms)
        return dataset
    
    @property
    def train_path(self):
        return os.path.join(self.save_path, 'train')
    
    @property
    def valid_path(self):
        return os.path.join(self.save_path, 'val')
    
    @property  # 이후에 dataset에 맞게 해줘야 할 수도 있긴한데, Train-Test 분포가 다르기도 하고 ImageNet 정도면 대표성을 띄고 있다고 생각하기 때문에 굳이 별도로 계산하지 않음
    def normalize(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):  # Default: ImageNet Config
        return transforms.Normalize(mean=mean, std=std)
    
    def build_train_transform(self, image_size=None, print_log=True):
        if image_size is None:
            image_size = self.image_size
        if print_log:
            print('Color jitter: %s, resize_scale: %s, img_size: %s' %
                  (self.distort_color, self.resize_scale, image_size))

        if self.distort_color == 'torch':
            color_transform = transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)
        elif self.distort_color == 'tf':
            color_transform = transforms.ColorJitter(brightness=32. / 255., saturation=0.5)
        else:
            color_transform = None
        
        # if isinstance(image_size, list):
        #     resize_transform_class = MyRandomResizedCrop
        #     print('Use MyRandomResizedCrop: %s, \t %s' % MyRandomResizedCrop.get_candidate_image_size(),
        #           'sync=%s, continuous=%s' % (MyRandomResizedCrop.SYNC_DISTRIBUTED, MyRandomResizedCrop.CONTINUOUS))
        # else:
        #     resize_transform_class = transforms.RandomResizedCrop

        train_transforms = [
            # resize_transform_class(image_size, scale=(self.resize_scale, 1.0)),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=(-90, 90)),
        ]
        if color_transform is not None:
            train_transforms.append(color_transform)
        train_transforms += [
            # transforms.ToTensor(),
            # self.normalize,
        ]

        train_transforms = transforms.Compose(train_transforms)
        return train_transforms

    def build_valid_transform(self, image_size=None):
        if image_size is None:
            image_size = self.active_img_size
        return transforms.Compose([
            # transforms.Resize(int(math.ceil(image_size / 0.875))),
            ModCrop(mod=4),
            # transforms.ToTensor(),
            # self.normalize,
        ])

    def assign_active_img_size(self, new_img_size):
        self.active_img_size = new_img_size
        if self.active_img_size not in self._valid_transform_dict:
            self._valid_transform_dict[self.active_img_size] = self.build_valid_transform()
        # change the transform of the valid and test set
        self.valid.dataset.transform = self._valid_transform_dict[self.active_img_size]
        self.test.dataset.transform = self._valid_transform_dict[self.active_img_size]
    
    def build_sub_train_loader(self, n_images, batch_size, num_worker=None, num_replicas=None, rank=None):
        # used for resetting running statistics
        if self.__dict__.get('sub_train_%d' % self.active_img_size, None) is None:
            if num_worker is None:
                num_worker = self.train.num_workers
            
            n_samples = len(self.train.dataset.samples)
            g = torch.Generator()
            g.manual_seed(DataProvider.SUB_SEED)
            rand_indexes = torch.randperm(n_samples, generator=g).tolist()
            
            new_train_dataset = self.train_dataset(
                self.build_train_transform(image_size=self.active_img_size, print_log=False))
            chosen_indexes = rand_indexes[:n_images]
            if num_replicas is not None:
                sub_sampler = MyDistributedSampler(new_train_dataset, num_replicas, rank, np.array(chosen_indexes))
            else:
                sub_sampler = torch.utils.data.sampler.SubsetRandomSampler(chosen_indexes)
            sub_data_loader = torch.utils.data.DataLoader(
                new_train_dataset, batch_size=batch_size, sampler=sub_sampler,
                num_workers=num_worker, pin_memory=True,
            )
            self.__dict__['sub_train_%d' % self.active_img_size] = []
            for images, labels in sub_data_loader:
                self.__dict__['sub_train_%d' % self.active_img_size].append((images, labels))
        return self.__dict__['sub_train_%d' % self.active_img_size]


################################################## SR 데이터셋. 기존 코드는 ImageFolder 사용해서 별도로 작성할 필요가 있었음
# import os
from PIL import Image
# import torchvision.transforms as transforms

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def get_image_paths(dir):
    assert os.path.isdir(dir), '%s is not a valid directory' % dir
    images = []
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    assert images, '%s has no valid image file' % dir
    return images

def get_image_paths_recursive(dir, images):
    assert os.path.isdir(dir), '%s is not a valid directory' % dir
    for root, subdirs, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                fpath = os.path.join(root, fname)
                images.append(fpath)
        for subdir in subdirs:
            subdir = os.path.join(root, subdir)
            get_image_paths_recursive(subdir, images)
    return images


# import numpy as np
# import torch
# import torch.utils.data as data
# import data.transforms as transforms

class Oracle_VideoDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): 이미지가 존재하는 디렉토리 경로
            transform (callable, optional): 샘플에 적용될 Optional transform
        """
        self.root_dir = root_dir
        self.transform = transform

        self.paths = []
        self.paths = sorted(get_image_paths_recursive(self.root_dir, self.paths))
        self.size = len(self.paths)

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        path = self.paths[index]
        H_img = self.transform(Image.open(path).convert('RGB'))
        L_img = get_transform_L()(H_img)
        H_tensor = transforms.ToTensor()(H_img)
        L_tensor = transforms.ToTensor()(L_img)
        out_dict = {'image': H_tensor, 'down_image': L_tensor}

        return out_dict


# from PIL import Image
# import torchvision.transforms as transforms
import random

def crop(img, i, j, h, w):
    """Crop the given PIL.Image.
    Args:
        img (PIL.Image): Image to be cropped.
        i: Upper pixel coordinate.
        j: Left pixel coordinate.
        h: Height of the cropped image.
        w: Width of the cropped image.
    Returns:
        PIL.Image: Cropped image.
    """
    return img.crop((j, i, j + w, i + h))

class ModCrop(object):
    """ModCrop the given PIL.Image.
    Args:
        mod (int): Crop to make the output size divisible by mod.
    """

    def __init__(self, mod):
        self.mod = int(mod)

    @staticmethod
    def get_params(img, mod):
        """Get parameters for ``crop`` for mod crop.
        Args:
            img (PIL.Image): Image to be cropped.
            mod (int): Crop to make the output size divisible by mod.
        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for mod crop.
        """
        w, h = img.size
        tw = w - w % mod
        th = h - h % mod
        return 0, 0, th, tw

    def __call__(self, img):
        """
        Args:
            img (PIL.Image): Image to be cropped.
        Returns:
            PIL.Image: Cropped image.
        """
        i, j, h, w = self.get_params(img, self.mod)
        return crop(img, i, j, h, w)

"""
Modified from the source "torchvision.transforms".
Use scale_factor as input instead of outputsize
"""
def scale(img, size, interpolation=Image.BICUBIC):
    assert isinstance(size, tuple) and len(size) == 2
    return img.resize(size[::-1], interpolation) # flip to (h,w) to (w,h)

class Scale(object):
    def __init__(self, scale_factor, interpolation=Image.BICUBIC):
        self.scale_factor = scale_factor
        self.interpolation = interpolation

    @staticmethod
    def get_params(img, scale_factor):
        w, h = img.size
        tw = int(w * scale_factor)
        th = int(h * scale_factor)
        return (th, tw)

    def __call__(self, img):
        size = self.get_params(img, self.scale_factor)
        return scale(img, size, self.interpolation)

def get_transform_L(opt=4):
    assert opt in (2, 4, 8)
    scale = 1 / opt
    transform_list = []
    transform_list.append(Scale(scale_factor=scale, interpolation=Image.BICUBIC))
    return transforms.Compose(transform_list)