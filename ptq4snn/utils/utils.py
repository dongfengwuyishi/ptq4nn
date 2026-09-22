# === 标准库 ===
import copy
import datetime
import logging
import math
import os
import random
import sys
import time
import warnings
from collections import defaultdict, deque
from typing import (
    Any,
    Callable,
    cast,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
)

# === 第三方库 ===
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import yaml
from easydict import EasyDict
from torch import Tensor
from torch.utils.data import DataLoader


logger = logging.getLogger('ptq4snn')


def parse_config(config_file):
    with open(config_file) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        cur_config = config
        cur_path = config_file
        while 'root' in cur_config:
            root_path = os.path.dirname(cur_path)
            cur_path = os.path.join(root_path, cur_config['root'])
            with open(cur_path) as r:
                root_config = yaml.load(r, Loader=yaml.FullLoader)
                for k, v in root_config.items():
                    if k not in config:
                        config[k] = v
                cur_config = root_config
    config = EasyDict(config)
    return config


def set_seed(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# hook function
class StopForwardException(Exception):
    """
    Used to throw and catch an exception to stop traversing the graph
    """
    pass


class DataSaverHook:
    """
    Forward hook that stores the input and output of a layer/block
    """
    def __init__(self, store_input=False, store_output=False, stop_forward=False):
        self.store_input = store_input
        self.store_output = store_output
        self.stop_forward = stop_forward

        self.input_store = None
        self.output_store = None

    def __call__(self, module, input_batch, output_batch):
        if self.store_input:
            self.input_store = input_batch
        if self.store_output:
            self.output_store = output_batch
        if self.stop_forward:
            raise StopForwardException


def init_logger(log_file_path: str, name: str = 'root') -> logging.Logger:
    LEVEL = logging.INFO
    logger = logging.getLogger(name)
    logger.setLevel(LEVEL)
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(LEVEL)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(LEVEL)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger

def timeExpand2d(inputs : torch.Tensor, time_steps = 4):
    r""""
        from (batch_size, channels, height, width) to (time_steps, batch_size, channels, height, width) by repeats for static image

        for dynamic image, already in time_steps format, directly return inputs
    """
    if inputs.dim() == 4: # (batch_size, channels, height, width) -> (time_steps, batch_size, channels, height, width)
        return inputs.unsqueeze(0).repeat(time_steps, 1, 1, 1, 1)
    elif inputs.dim() == 5:
        return inputs # already in time_steps format
    else:
        raise ValueError(f'Invalid input shape {inputs.shape}, expected 4D or 5D tensor, but got {inputs.dim()}D tensor')

class Dim1ToDim0Wrapper:
    __slots__ = ("_t", "_ndim")

    def __init__(self, tensor: torch.Tensor):
        if tensor.ndim < 2:
            raise ValueError("需要至少 2 维的 Tensor")
        self._t = tensor
        self._ndim = tensor.ndim

    # -------- 关键协议 --------
    def __getitem__(self, idx):
        # 把 t[i] 变成 t[:, i]
        if isinstance(idx, tuple):
            # 用户传了多个切片，如 t[1, 2]
            return self._t[(slice(None),) + idx]
        else:
            # 单个索引/切片，如 t[3], t[:5]
            return self._t[:, idx]

    def __setitem__(self, idx, value):
        if isinstance(idx, tuple):
            self._t[(slice(None),) + idx] = value
        else:
            self._t[:, idx] = value

    # -------- 让对象更像一个“张量” --------
    def __len__(self):
        # 返回第二维长度
        return self._t.shape[1]

    # size
    def size(self, dim=None):
        if dim is None:
            return self.shape
        else:
            return self.shape[dim]

    @property
    def shape(self):
        # 把 (B, L, ...) 变成 (L, B, ...)
        return (self._t.shape[1], self._t.shape[0]) + self._t.shape[2:]

    def numpy(self):
        return self._t.numpy()

    def __repr__(self):
        return repr(self._t[:, :])  # 按新的维度顺序打印

class CriterionWrapper(nn.CrossEntropyLoss):
    r"""
        input: (time_steps, batch_size, num_classes)
        target: (batch_size)

    """

    def __init__(self, weight: Optional[Tensor] = None, size_average=None, ignore_index: int = -100,reduce=None, reduction: str = 'mean', label_smoothing: float = 0.0, lamb = 0.05, tau = 1.0, is_tet = True):
        super(CriterionWrapper, self).__init__(weight, size_average, ignore_index, reduce, reduction, label_smoothing)
        self.lamb = lamb
        self.tau = tau
        self.is_tet = is_tet

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        if self.is_tet:
            T = input.size(0)
            device = input.device
            loss_es = 0
            mmd_y = torch.full_like(input, self.tau, device=device)
            for t in range(T):
                loss_es += super().forward(input[t], target)
            loss_es /= T
            if self.lamb != 0:
                MMDLoss = torch.nn.MSELoss()
                loss_mmd = MMDLoss(input, mmd_y)
            else:
                loss_mmd = 0
            return (1 - self.lamb) * loss_es + self.lamb * loss_mmd
        else:
            return super().forward(input.mean(0), target)

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    if output.dim() == 3:
        output = output.sum(dim=0) # (time_steps, batch_size, num_classes) -> (batch_size, num_classes)
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target[None])

        res = []
        for k in topk:
            correct_k = correct[:k].flatten().sum(dtype=torch.float32)
            res.append(correct_k * (100.0 / batch_size))
        return res

class WarmupWarper:
    def __init__(self, warmup_epochs, max_lr, start_lr, after_scheduler = None, optimizer = None, now_epoch = 0):
        self.warmup_epochs = warmup_epochs
        self.current_steps = now_epoch
        self.max_lr = max_lr
        self.start_lr = start_lr
        assert after_scheduler is not None, "after_scheduler should not be None"
        self.after_scheduler = after_scheduler
        assert optimizer is not None, "optimizer should not be None"
        self.optimizer = optimizer
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.start_lr


    def step(self):
        self.current_steps += 1
        if self.current_steps <= self.warmup_epochs:
            lr = self.start_lr + (self.max_lr - self.start_lr) * self.current_steps / self.warmup_epochs
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        else:
            self.after_scheduler.step()


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()

def is_main_process():
    return get_rank() == 0

class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)

class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None, logger=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        if torch.cuda.is_available():
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'iter time: {time}',
                'data: {data}',
                'max mem: {memory:.0f}'
            ])
        else:
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'iter time: {time}',
                'data: {data}'
            ])
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == print_freq - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    if logger is not None:
                        logger.info(log_msg.format(
                            i, len(iterable), eta=eta_string,
                            meters=str(self),
                            time=str(iter_time), data=str(data_time),
                            memory=torch.cuda.max_memory_allocated() / MB))
                    else:
                        print(log_msg.format(
                            i, len(iterable), eta=eta_string,
                            meters=str(self),
                            time=str(iter_time), data=str(data_time),
                            memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    if logger is not None:
                        logger.info(log_msg.format(
                            i, len(iterable), eta=eta_string,
                            meters=str(self),
                            time=str(iter_time), data=str(data_time)))
                    else:
                        print(log_msg.format(
                            i, len(iterable), eta=eta_string,
                            meters=str(self),
                            time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        if logger is not None:
            logger.info('{} Total time: {}'.format(header, total_time_str))
        else:
            print('{} Total time: {}'.format(header, total_time_str))

def train_one_epoch(data_loader : DataLoader, model : nn.Module, criterion : CriterionWrapper, optimizer : optim.Optimizer, epoch : int, total_epochs : int, device : torch.device, logger : logging.Logger, log_interval=100):
    model.to(device)
    model.train()

    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value}'))
    metric_logger.add_meter('img/s', SmoothedValue(window_size=10, fmt='{value:.1f}'))

    header = 'Epoch: [{}/{}]'.format(epoch, total_epochs)

    for image, target in metric_logger.log_every(data_loader, log_interval, header, logger):
        start_time = time.time()
        image, target = image.to(device), target.to(device)
        output = model(image)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        batch_size = target.shape[0]
        loss_s = loss.item()
        if math.isnan(loss_s):
            raise ValueError('loss is Nan')
        acc1_s = acc1.item()
        acc5_s = acc5.item()

        metric_logger.update(loss=loss_s, lr=optimizer.param_groups[0]["lr"])

        metric_logger.meters['acc1'].update(acc1_s, n=batch_size)
        metric_logger.meters['acc5'].update(acc5_s, n=batch_size)
        metric_logger.meters['img/s'].update(batch_size / (time.time() - start_time))

    metric_logger.synchronize_between_processes()
    if is_main_process():
        logger.info(f' * Train Acc@1 = {metric_logger.acc1.global_avg:.3f}, Acc@5 = {metric_logger.acc5.global_avg:.3f}, train loss = {metric_logger.loss.global_avg:.5f}')
    return metric_logger.loss.global_avg, metric_logger.acc1.global_avg, metric_logger.acc5.global_avg


@torch.no_grad()
def val(data_loader : DataLoader, model : nn.Module, criterion : CriterionWrapper, device : torch.device, logger : logging.Logger, desc : str = None, log_interval=100, header='Test:'):
    model.to(device)
    model.eval()
    metric_logger = MetricLogger(delimiter="  ")
    with torch.no_grad():
        for image, target in metric_logger.log_every(data_loader, log_interval, header, logger):
            image = image.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            output = model(image)
            loss = criterion(output, target)

            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            # FIXME need to take into account that the datasets
            # could have been padded in distributed setup
            batch_size = target.shape[0]
            metric_logger.update(loss=loss.item())
            metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
            metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()

    loss, acc1, acc5 = metric_logger.loss.global_avg, metric_logger.acc1.global_avg, metric_logger.acc5.global_avg
    if is_main_process():
        logger.info(f' * Test Acc@1 = {acc1:.3f}, Acc@5 = {acc5:.3f}, test loss = {loss:.5f}')
    return loss, acc1, acc5


def pre_save_process(model : nn.Module):
    # if in dp wrapper, unwrap it
    if isinstance(model, torch.nn.parallel.DataParallel) or isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model = copy.deepcopy(model.module)
    else:
        model = copy.deepcopy(model)
    from ptq4snn.model.common import reset_net
    reset_net(model)
    model.to('cpu')
    return model
