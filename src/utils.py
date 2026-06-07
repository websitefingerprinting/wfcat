from __future__ import annotations

import os
import random
from pathlib import Path
from time import time
from typing import Callable, List, Tuple, Union

import numpy as np
import torch

from features import parse_trace

PR_THRES_NUM = 10


def seed_everything(seed: int = 42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def increment_path(path: os.PathLike, exist_ok: bool = False, sep: str = '', mkdir: bool = False) -> Path:
    path = Path(path)
    if path.exists() and not exist_ok:
        path, suffix = (path.with_suffix(''), path.suffix) if path.is_file() else (path, '')
        for n in range(2, 9999):
            p = f'{path}{sep}{n}{suffix}'
            if not os.path.exists(p):
                break
        path = Path(p)
    if mkdir:
        path.mkdir(parents=True, exist_ok=True)
    return path


def timeit(f: Callable):
    def wrap(*args, **kwargs):
        ts = time()
        result = f(*args, **kwargs)
        te = time()
        print('func:%r took: %2.4f sec' % (f.__name__, te - ts))
        return result
    return wrap


def get_flist_label_single_domain(data_path: Union[str, os.PathLike], mon_cls: int, mon_inst: int,
                                  unmon_inst: int, suffix: str = '.cell') -> Tuple[np.ndarray, np.ndarray]:
    flist = []
    labels = []
    for cls in range(mon_cls):
        for inst in range(mon_inst):
            path = os.path.join(data_path, f'{cls}-{inst}{suffix}')
            if os.path.exists(path):
                flist.append(path)
                labels.append(cls)
    for inst in range(unmon_inst):
        path = os.path.join(data_path, f'{inst}{suffix}')
        if os.path.exists(path):
            flist.append(path)
            labels.append(mon_cls)
    assert len(flist) > 0, f"No files found in {data_path}!"
    return np.array(flist), np.array(labels)


def get_flist_label_multi_domain(data_path: Union[str, os.PathLike], mon_cls: int, mon_inst: int,
                                 unmon_inst: int, page_per_class: int = 1,
                                 suffix: str = '.cell') -> Tuple[int, np.ndarray, np.ndarray]:
    flist = []
    labels = []
    max_mon_label = -1
    for cls in range(mon_cls):
        for inst in range(mon_inst):
            path = os.path.join(data_path, f'{cls}-{inst}{suffix}')
            if os.path.exists(path):
                label = cls // page_per_class
                flist.append(path)
                labels.append(label)
                max_mon_label = max(max_mon_label, label)

    for inst in range(unmon_inst):
        path = os.path.join(data_path, f'{inst}{suffix}')
        if os.path.exists(path):
            flist.append(path)
            labels.append(max_mon_label + 1)

    assert len(flist) > 0, f"No files found in {data_path}!"
    return max_mon_label + 1, np.array(flist), np.array(labels)


def return_loading_bandwidth(path: Union[str, os.PathLike]) -> float:
    trace = parse_trace(path)
    return len(trace) / (float(trace[-1, 0]) + 1e-5)


def select_fast_slow(flist: Union[List, np.ndarray], labels: Union[List, np.ndarray],
                     train_mode: str = 'fast', ratio: float = 0.9) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_list, test_list = [], []
    train_labels, test_labels = [], []

    for label in np.unique(labels):
        class_files = flist[labels == label]
        bandwidths = np.array([return_loading_bandwidth(f) for f in class_files])
        indices = np.argsort(-bandwidths) if train_mode == 'fast' else np.argsort(bandwidths)
        n_train = int(len(indices) * ratio)
        train_list.extend(class_files[indices[:n_train]])
        train_labels.extend([label] * n_train)
        test_list.extend(class_files[indices[n_train:]])
        test_labels.extend([label] * (len(indices) - n_train))

    return np.array(train_list), np.array(train_labels), np.array(test_list), np.array(test_labels)


def get_grad_norm(model):
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm ** 0.5
