import numpy as np
from typing import Optional, Callable
import os
from torchvision import datasets
from torchvision import transforms
from torchvision.transforms import Resize, Normalize
import torch
from torchvision.datasets.cifar import CIFAR10
from torchvision.datasets.cifar import CIFAR100
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from torchvision.transforms import v2
import random

nclass_and_in_channels_dict = {
    # 'name': (num_classes, num_channels)
    'cifar10': (10, 3),
    'cifar100': (100, 3),
    'cifar10dvs': (10, 2),
    'imagenet': (1000, 3),
}

allowed_datasets = nclass_and_in_channels_dict.keys()


def load_npz_frames(file_name: str) -> np.ndarray:
    return np.load(file_name, allow_pickle=True)['frames'].astype(np.float32)

class MyCIFAR10DVS(datasets.DatasetFolder):
    def __init__(self,
        root: str,
        loader = load_npz_frames,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        train = None
        ):
        if train is not None:
            if train:
                root = os.path.join(root, 'train')
            else:
                root = os.path.join(root, 'test')
        super().__init__(root=root, loader=loader, extensions=('.npz', ), transform=transform,
                         target_transform=target_transform)

class MyDVSGesture(datasets.DatasetFolder):
    def __init__(self,
        root: str,
        loader = load_npz_frames,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        train = None
        ):
        if train is not None:
            if train:
                root = os.path.join(root, 'train')
            else:
                root = os.path.join(root, 'test')
        super().__init__(root=root, loader=loader, extensions=('.npz', ), transform=transform,
                         target_transform=target_transform)

class StratifiedSampler(torch.utils.data.Sampler):
    def __init__(self, stratified_categories_index, num_samples_per_category = 3):
        super(StratifiedSampler, self).__init__(object())
        self.stratified_categories_index = stratified_categories_index
        self.num_samples_per_category = num_samples_per_category

        self.lengths = [num_samples_per_category for _ in self.stratified_categories_index]
        self.size = sum(self.lengths)

    def __iter__(self):
        to_sample_list = []
        for length, stratified_category in zip(self.lengths, self.stratified_categories_index):
            to_sample_list += random.sample(stratified_category, length)
        random.shuffle(to_sample_list)
        return iter(to_sample_list)

    def __len__(self):
        return self.size


def get_stratified_categories_index(dataset):
    stratified_categories_index = [list() for _ in range(1000)]
    for index, (path, category) in enumerate(dataset.samples):
        stratified_categories_index[category].append(index)
    return stratified_categories_index


def get_datasets(dataset_name: str, data_root: str, data_aug = False, mixup = False, cutmix = False, random_erase = False, distributed = False):
    dataset_name = dataset_name.lower()
    collate_fn = None
    if dataset_name == 'cifar10':
        train_transform_list = []
        train_transform_list += [transforms.RandomCrop(32, padding=4)]
        train_transform_list += [transforms.RandomHorizontalFlip()]
        if data_aug:
            train_transform_list += [transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.CIFAR10)]
        train_transform_list += [transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]
        if random_erase:
            train_transform_list += [v2.RandomErasing()]
        train_transform = transforms.Compose(train_transform_list)
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        train_dataset = CIFAR10(root=data_root, train=True, download=True, transform=train_transform)
        train4test_dataset = CIFAR10(root=data_root, train=True, download=True, transform=test_transform)
        test_dataset = CIFAR10(root=data_root, train=False, download=True, transform=test_transform)
        in_channels = 3
        nclass = 10
    elif dataset_name == 'cifar100':
        train_transform_list = [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
        ]
        if data_aug:
            train_transform_list += [transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.CIFAR10)]
        train_transform_list += [
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ]
        if random_erase:
            train_transform_list += [v2.RandomErasing()]
        train_transform = transforms.Compose(train_transform_list)
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])
        train_dataset = CIFAR100(root=data_root, train=True, download=False, transform=train_transform)
        train4test_dataset = CIFAR100(root=data_root, train=True, download=False, transform=test_transform)
        test_dataset = CIFAR100(root=data_root, train=False, download=False, transform=test_transform)
        in_channels = 3
        nclass = 100
    elif dataset_name == 'cifar10dvs':
        data_root = os.path.join(data_root, 'cifar10dvs', 'frames_number_10_split_by_number')
        train_transform_list = []
        train_transform_list += [lambda x: torch.from_numpy(x)]
        train_transform_list += [transforms.Resize(48)]
        train_transform_list += [transforms.RandomCrop(48, padding=4)]
        train_transform_list += [v2.RandomErasing()]
        train_transform = transforms.Compose(train_transform_list)
        test_transform = transforms.Compose([
            lambda x: torch.from_numpy(x),
            transforms.Resize(48),
        ])
        train_dataset = MyCIFAR10DVS(root=data_root, train=True, transform=train_transform)
        train4test_dataset = MyCIFAR10DVS(root=data_root, train=True, transform=test_transform)
        test_dataset = MyCIFAR10DVS(root=data_root, train=False, transform=test_transform)
        in_channels = 2
        nclass = 10
    elif dataset_name == 'imagenet':
        data_root = os.path.join(data_root, 'imagenet')
        train_transform_list = []
        train_transform_list += [transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip()]
        if data_aug:
            train_transform_list += [transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.IMAGENET)]
        train_transform_list += [transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]
        if random_erase:
            train_transform_list += [v2.RandomErasing()]
        train_transform = transforms.Compose(train_transform_list)
        test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        train_dir = os.path.join(data_root, 'train')
        test_dir = os.path.join(data_root, 'val')
        train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transform)
        train4test_dataset = datasets.ImageFolder(root=train_dir, transform=test_transform)
        test_dataset = datasets.ImageFolder(root=test_dir, transform=test_transform)
        in_channels = 3
        nclass = 1000
    else:
        raise ValueError(f'Dataset {dataset_name} not supported, only support {allowed_datasets}')
    if mixup or cutmix:
        cutmix = v2.CutMix(num_classes=nclass)
        mixup = v2.MixUp(num_classes=nclass)
        if cutmix and mixup:
            func = v2.RandomChoice([cutmix, mixup])
        elif cutmix:
            func = cutmix
        elif mixup:
            func = mixup
        def collate_fn(batch):
            return func(*default_collate(batch))

    if distributed:
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
        train4test_sampler = DistributedSampler(train4test_dataset, shuffle=False)
        test_sampler = DistributedSampler(test_dataset, shuffle=False)
    else:
        train_sampler = RandomSampler(train_dataset)
        train4test_sampler = RandomSampler(train4test_dataset)
        test_sampler = RandomSampler(test_dataset)

    return train_dataset, train4test_dataset, test_dataset, train_sampler, train4test_sampler, test_sampler, nclass, in_channels, collate_fn

if __name__ == '__main__':
    pass
